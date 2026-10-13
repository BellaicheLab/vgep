import os
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import mudata
from vgep import spatial_transcriptome
from vgep import plotter
from vgep.grn.network import create_tf_cluster_network


def test_generate_vgep(sample_data, temp_output_path):
    """Test the spatial transcriptome generation function."""
    # Extract data from the fixture
    atlas = sample_data["atlas"]
    validation = sample_data["validation"]
    geometry = sample_data["geometry"]
    raw_count = sample_data["raw_count"]
    macrochaete = sample_data["macrochaete"]

    # Generate the spatial transcriptome
    transcriptome = spatial_transcriptome.generate_vgep(
        sc_data=raw_count,
        atlas=atlas,
        geometry=geometry,
        cutoff=0.98,
        validation=validation,
        morphological_marks=macrochaete,
    )

    # Test that a mudata object is returned
    assert isinstance(transcriptome, mudata.MuData)

    # Test that the mudata object has the expected modalities
    assert "st" in transcriptome.mod

    # Test that spatial coordinates are present in obs
    assert "x" in transcriptome["st"].obs.columns
    assert "y" in transcriptome["st"].obs.columns

    # Test that results recap is in uns
    assert "results_recap" in transcriptome["st"].uns

    # Test correlations are present
    assert "correlations_genes_training" in transcriptome["st"].uns
    assert "correlations_genes_validation" in transcriptome["st"].uns

    # Test that the output can be saved
    output_path = os.path.join(temp_output_path, "test_st.h5mu")
    transcriptome.write(output_path)
    assert os.path.exists(output_path)

    # Try to load it back to ensure it was saved correctly
    loaded = mudata.read(output_path)
    assert isinstance(loaded, mudata.MuData)


def test_plot_recap(sample_data, monkeypatch):
    """Test the plot_recap function with plt.show mocked to not display."""
    # Suppress actual plotting
    monkeypatch.setattr(plt, "show", lambda: None)

    # Generate a spatial transcriptome
    transcriptome = spatial_transcriptome.generate_vgep(
        sc_data=sample_data["raw_count"],
        atlas=sample_data["atlas"],
        geometry=sample_data["geometry"],
        cutoff=0.98,
        validation=sample_data["validation"],
        morphological_marks=sample_data["macrochaete"],
    )

    # Plot recap and ensure it doesn't raise exceptions
    fig = plotter.plot_recap(
        transcriptome, sample_data["atlas"], sample_data["validation"]
    )

    # Basic check that a figure was created
    assert isinstance(fig, plt.Figure)


def test_specific_genes_and_recap_after_reload(sample_data, tmp_path):
    """specific_genes subsets the projection; the recap works on a reloaded h5mu without landmarks."""
    genes = list(sample_data["atlas"].columns) + list(sample_data["validation"].columns)
    transcriptome = spatial_transcriptome.generate_vgep(
        sc_data=sample_data["raw_count"],
        atlas=sample_data["atlas"],
        geometry=sample_data["geometry"],
        cutoff=0.98,
        validation=sample_data["validation"],
        specific_genes=genes,
    )
    assert transcriptome["st"].n_vars == len(genes)

    transcriptome.write(tmp_path / "transcriptome.h5mu")
    transcriptome = mudata.read_h5mu(tmp_path / "transcriptome.h5mu")
    fig = plotter.plot_recap(
        transcriptome, sample_data["atlas"], sample_data["validation"], include_notum_expression=False
    )
    assert isinstance(fig, plt.Figure)


def test_tf_cluster_network_without_tau(tmp_path):
    """A TF missing from tau_scores must not break the GraphML export."""
    combinations = pd.DataFrame({"0": [1.0, 0.0], "1": [1.0, 1.0]}, index=["tfA", "tfB"])
    create_tf_cluster_network(combinations, pd.Series({"tfA": 0.9}), output_path=str(tmp_path / "net"))
