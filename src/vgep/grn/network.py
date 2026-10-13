"""GRN graph construction, clustering, and Cytoscape export."""

import glob
import json
import logging
from pathlib import Path

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import cytoscape_data
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fetch_adjacency_file_paths(folder_path):
    """Fetch all TSV file paths in the specified folder.

    Parameters
    ----------
    folder_path : str or Path
        The path to the folder where the files are located.

    Returns
    -------
    list
        A list of paths to files matching the specified pattern.
    """
    # Define the pattern to match files of interest
    pattern = f"{folder_path}/*.tsv"

    # Use glob to find files matching the pattern
    file_paths = glob.glob(pattern)

    return file_paths


def load_all_adjacencies(adjacencies_paths):
    """Load all adjacency TSV files into a single DataFrame.

    Parameters
    ----------
    adjacencies_paths : list of str
        Paths to the TSV files containing adjacencies.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all adjacency data with a 'source_file' column.
    """
    logger.info("Loading %d adjacency files into memory...", len(adjacencies_paths))

    all_adjacencies = []

    # Load all files with a progress bar
    for i, path in enumerate(tqdm(adjacencies_paths, desc="Loading adjacency files")):
        # Load the current TSV file
        data = pd.read_csv(path, sep="\t", header=0)
        # Add source file index for tracking
        data["source_file"] = i
        all_adjacencies.append(data)

    # Combine all files
    combined_data = pd.concat(all_adjacencies, ignore_index=True)

    # Count total unique TFs and files for statistics
    total_tfs = combined_data["TF"].nunique()
    total_files = len(adjacencies_paths)

    logger.info("Loaded data for %d TFs from %d files", total_tfs, total_files)
    logger.debug("Total adjacency records: %d", len(combined_data))

    return combined_data


def get_tf_targets(
    combined_adjacencies, tf_name, min_occurrence_fraction=0.8, n_top=10
):
    """Get top targets for a single TF with consistency filtering.

    Parameters
    ----------
    combined_adjacencies : pd.DataFrame
        Pre-loaded adjacency data.
    tf_name : str
        The name of the TF to filter by.
    min_occurrence_fraction : float
        Minimum fraction of runs a target must appear in (default 0.8).
    n_top : int
        The number of top targets to return based on average importance.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top n_top targets and their average importance.
    """
    # Filter by TF name
    tf_data = combined_adjacencies[combined_adjacencies["TF"] == tf_name]

    if tf_data.empty:
        logger.warning("No data found for TF: %s", tf_name)
        return pd.DataFrame(columns=["target", "importance", "occurrence_fraction"])

    # Get total number of source files
    total_files = combined_adjacencies["source_file"].nunique()

    # Count occurrences of each target across different source files
    target_occurrences = tf_data.groupby("target")["source_file"].nunique()

    # Filter targets that appear in at least min_occurrence_fraction of files
    min_occurrences = int(total_files * min_occurrence_fraction)
    consistent_targets = target_occurrences[target_occurrences >= min_occurrences].index

    if len(consistent_targets) == 0:
        logger.warning(
            "No targets found in at least %.0f%% of runs for TF: %s",
            min_occurrence_fraction * 100, tf_name,
        )
        return pd.DataFrame(columns=["target", "importance", "occurrence_fraction"])

    # Filter data to include only consistent targets
    filtered_tf_data = tf_data[tf_data["target"].isin(consistent_targets)]

    # Group by target and compute the mean importance
    mean_importance = (
        filtered_tf_data.groupby("target")["importance"].mean().reset_index()
    )

    # Add occurrence fraction information
    mean_importance["occurrence_fraction"] = mean_importance["target"].apply(
        lambda t: target_occurrences[t] / total_files
    )

    # Sort by importance and take top n_top
    top_targets = mean_importance.sort_values(by="importance", ascending=False).head(
        n_top
    )

    return top_targets


def create_tf_target_network(
    combined_adjacencies,
    tf_list,
    total_tf_list,
    cluster_name,
    min_occurrence_fraction=0.8,
    n_top=10,
):
    """Create a GRN graph with cluster, TF, and target gene nodes using pre-loaded adjacency data.

    Parameters
    ----------
    combined_adjacencies : pd.DataFrame
        Pre-loaded adjacency data.
    tf_list : list of str
        The list of TF names to include in the graph.
    total_tf_list : list
        All the annotated TFs.
    cluster_name : str
        The name of the central cluster node.
    min_occurrence_fraction : float
        Minimum fraction of runs a target must appear in.
    n_top : int
        The number of top targets to return for each TF.

    Returns
    -------
    nx.Graph
        An undirected graph of the network.
    """
    # Initialize an undirected graph
    G = nx.Graph()

    # Add the cluster node with its type
    G.add_node(cluster_name, node_type="cluster")

    # First, add all TFs to the graph
    for tf in tf_list:
        G.add_node(tf, node_type="tf")
        # Cluster-TF edges represent specificity (from tau-based assignment)
        G.add_edge(cluster_name, tf, weight=1, edge_type="specificity")

    # Then, for each TF, find top targets and link them appropriately
    for tf in tqdm(tf_list, desc=f"Building network for cluster {cluster_name}"):
        all_targets = get_tf_targets(
            combined_adjacencies,
            tf_name=tf,
            min_occurrence_fraction=min_occurrence_fraction,
            n_top=4501,
        )

        top_targets = all_targets.head(n_top)

        for _, row in top_targets.iterrows():
            target, importance = row["target"], row["importance"]
            occurrence = row.get(
                "occurrence_fraction", 1.0
            )  # Default to 1.0 if not available

            # Check if the target is also a TF
            if target in tf_list:
                # TF-TF or TF-target edges from GRN adjacency represent regulation
                G.add_edge(
                    tf,
                    target,
                    weight=importance,
                    occurrence=occurrence,
                    edge_type="regulation",
                )
            else:
                # Add target as a new node if not already present and link it to the TF
                if not G.has_node(target):
                    if target in total_tf_list:
                        G.add_node(target, node_type="tf")
                    else:
                        G.add_node(target, node_type="target")

                G.add_edge(
                    tf,
                    target,
                    weight=importance,
                    occurrence=occurrence,
                    edge_type="regulation",
                )

    return G


def annotate_network_with_leiden_clusters(G, resolution=1.0, n_iterations=-1):
    """Annotate the network with cluster IDs using the Leiden algorithm.

    Modifies the input graph G in-place by adding a 'cluster' attribute to each node.

    Parameters
    ----------
    G : nx.Graph
        The graph to be clustered.
    resolution : float
        The resolution parameter for the Leiden algorithm.
    n_iterations : int
        Number of iterations to perform; -1 for the algorithm to automatically determine.
    """
    # Before converting G, assign the name attribute to each node
    for node in G.nodes():
        G.nodes[node]["name"] = node

    # Now convert
    ig_G = ig.Graph.from_networkx(G)

    # Apply the Leiden algorithm using a partition type that accepts resolution_parameter
    partition = leidenalg.find_partition(
        ig_G,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
    )

    # Proceed with Leiden clustering: retrieve the original node id from the name attribute.
    for idx, cluster in enumerate(partition):
        for node in cluster:
            original_node_id = ig_G.vs[node]["name"]
            G.nodes[original_node_id]["cluster"] = idx


def generate_grn_graphs_optimized(
    combinations_df,
    combined_adjacencies,
    tf_list,
    min_occurrence_fraction=0.8,
    n_top=10,
    z_cluster_means=None,
):
    """Generate Gene Regulatory Network graphs for each cluster using pre-loaded adjacency data.

    Parameters
    ----------
    combinations_df : pd.DataFrame
        DataFrame with binary indicators of TF specificity to clusters.
    combined_adjacencies : pd.DataFrame
        Pre-loaded combined adjacency data.
    tf_list : list
        List of all transcription factors.
    min_occurrence_fraction : float
        Minimum fraction of runs a target must appear in.
    n_top : int
        Number of top targets to include for each TF.
    z_cluster_means : pd.DataFrame or None
        Optional genes x clusters DataFrame of mean z-scores.

    Returns
    -------
    dict
        Dictionary mapping cluster names to their GRN graphs.
    """
    # Dictionary to store graphs for each cluster
    cluster_graphs = {}

    # For each cluster
    for cluster in combinations_df.columns:
        # Get TFs specific to this cluster
        cluster_tfs = combinations_df[combinations_df[cluster] == 1].index.tolist()

        if not cluster_tfs:
            logger.warning("No specific TFs found for cluster %s, skipping", cluster)
            continue

        logger.info("Creating GRN for cluster %s with %d specific TFs", cluster, len(cluster_tfs))

        # Create network using pre-loaded data
        G = create_tf_target_network(
            combined_adjacencies=combined_adjacencies,
            tf_list=cluster_tfs,
            total_tf_list=tf_list,
            cluster_name=f"Cluster_{cluster}",
            min_occurrence_fraction=min_occurrence_fraction,
            n_top=n_top,
        )

        # Annotate nodes with per-cluster z-normalized expression if provided
        if z_cluster_means is not None and cluster in z_cluster_means.columns:
            for node, attrs in G.nodes(data=True):
                # Only annotate gene-like nodes (TFs and targets)
                if attrs.get("node_type") in ("tf", "target") and node in z_cluster_means.index:
                    try:
                        G.nodes[node]["z_expr"] = float(z_cluster_means.at[node, cluster])
                    except Exception:
                        G.nodes[node]["z_expr"] = np.nan

        cluster_graphs[cluster] = G

        # Log basic network statistics
        logger.debug(
            "  Network has %d nodes and %d edges",
            G.number_of_nodes(), G.number_of_edges(),
        )
        n_seeds = len(cluster_tfs)
        n_genes = max(0, G.number_of_nodes() - 1)
        logger.debug("  Total TF seeds: %d; Total target genes: %d", n_seeds, n_genes)

    return cluster_graphs


def export_network_for_cytoscape(G, output_folder, name, tau_scores=None):
    """Export a network in formats compatible with Cytoscape.

    Parameters
    ----------
    G : nx.Graph
        The graph to export.
    output_folder : str or Path
        Folder where to save the exported files.
    name : str
        Base name for the exported files.
    tau_scores : pd.Series, optional
        Optional Series of tau scores indexed by gene names to be added as a node attribute.

    Returns
    -------
    dict
        Dictionary with paths to exported files.
    """
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # If provided, annotate nodes with tau scores (where available)
    if tau_scores is not None:
        tau_index = set(tau_scores.index)
        for node in G.nodes():
            if node in tau_index and "tau" not in G.nodes[node]:
                try:
                    G.nodes[node]["tau"] = float(tau_scores.loc[node])
                except Exception:
                    G.nodes[node]["tau"] = np.nan

    # Export as GraphML (preserves all node and edge attributes)
    graphml_path = output_folder / f"{name}.graphml"
    nx.write_graphml(G, graphml_path)

    # Export as Cytoscape JSON format
    cyjs_data = cytoscape_data(G)
    cyjs_path = output_folder / f"{name}.cyjs"
    with open(cyjs_path, "w") as f:
        json.dump(cyjs_data, f, indent=2)

    # Export as edge list for simple import
    edgelist_path = output_folder / f"{name}.txt"
    nx.write_edgelist(G, edgelist_path, data=True)

    # Additionally export tidy node/edge DataFrames (CSV) for convenient downstream use
    # Nodes dataframe
    nodes_records = []
    for node, attrs in G.nodes(data=True):
        nodes_records.append(
            {
                "id": node,
                "node_type": attrs.get("node_type"),
                "cluster": attrs.get("cluster"),
                "tau": attrs.get("tau"),
                "z_expr": attrs.get("z_expr"),
                "degree": int(G.degree(node)),
            }
        )
    nodes_df = pd.DataFrame(nodes_records)
    nodes_csv_path = output_folder / f"{name}_nodes.csv"
    nodes_df.to_csv(nodes_csv_path, index=False)

    # Edges dataframe
    edges_records = []
    for u, v, data in G.edges(data=True):
        edges_records.append(
            {
                "source": u,
                "target": v,
                "weight": data.get("weight"),
                "occurrence": data.get("occurrence"),
                "edge_type": data.get("edge_type"),
            }
        )
    edges_df = pd.DataFrame(edges_records)
    edges_csv_path = output_folder / f"{name}_edges.csv"
    edges_df.to_csv(edges_csv_path, index=False)

    logger.debug("Exported network to: %s, %s, %s", graphml_path, cyjs_path, edgelist_path)

    return {"graphml": graphml_path, "cyjs": cyjs_path, "edgelist": edgelist_path}


def create_tf_cluster_network(
    combinations_df, tau_scores, output_path="tf_cluster_network"
):
    """Create a network with clusters and their associated TFs, with tau scores as attributes.

    Removes any isolated nodes with no connections.

    Parameters
    ----------
    combinations_df : pd.DataFrame
        DataFrame where rows are TFs, columns are clusters, and values are binary (1.0/0.0).
    tau_scores : pd.Series
        Series of tau scores with TF names as index.
    output_path : str
        Base path for output files (no extension).

    Returns
    -------
    tuple of (nx.Graph, dict)
        The created network graph and dictionary with paths to exported files.
    """
    # Initialize graph
    G = nx.Graph()

    # Add cluster nodes
    for cluster in combinations_df.columns:
        cluster_id = f"Cluster_{cluster}"
        G.add_node(cluster_id, node_type="cluster")

    # Add TF nodes with tau scores and connect them to clusters
    for tf, row in combinations_df.iterrows():
        # GraphML cannot store None, so a TF without a tau score gets NaN
        G.add_node(tf, node_type="tf", tau=float(tau_scores.get(tf, np.nan)))

        # Connect TF to its clusters
        for cluster, value in row.items():
            if value == 1.0:
                # These edges represent specificity of TFs to clusters
                G.add_edge(tf, f"Cluster_{cluster}", edge_type="specificity")

    # Remove isolated nodes (nodes with no connections)
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        logger.debug("Removing %d isolated nodes with no connections", len(isolated_nodes))
        G.remove_nodes_from(isolated_nodes)

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # Export in multiple formats
    export_paths = {}

    # GraphML format (preserves all node and edge attributes)
    graphml_path = f"{output_path}.graphml"
    nx.write_graphml(G, graphml_path)
    export_paths["graphml"] = graphml_path

    # Cytoscape JSON format
    cyjs_data = cytoscape_data(G)
    cyjs_path = f"{output_path}.cyjs"
    with open(cyjs_path, "w") as f:
        json.dump(cyjs_data, f, indent=2)
    export_paths["cyjs"] = cyjs_path

    # Edge list for simple import
    edgelist_path = f"{output_path}.txt"
    nx.write_edgelist(G, edgelist_path, data=True)
    export_paths["edgelist"] = edgelist_path

    logger.debug("Network exported to: %s, %s, %s", graphml_path, cyjs_path, edgelist_path)

    return G, export_paths
