import tempfile
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_gene_names():
    """Sample gene names for testing."""
    training_genes = ["al", "ara", "B-H1", "bi", "Diap1", "esg", "exd", "eyg"]
    validation_genes = ["C15", "caup", "crp", "Dl", "dpp", "fj", "hbs"]
    return {"training": training_genes, "validation": validation_genes}


@pytest.fixture
def sample_data(sample_gene_names, tmp_path):
    """Create sample data for testing spatial transcriptome generation."""
    # Number of spatial spots and cells
    n_spots = 50
    n_cells = 100

    # Create sample coordinates for spatial data
    x_coords = np.random.randint(1, 20, n_spots)
    y_coords = np.random.randint(1, 20, n_spots)

    # Create geometry dataframe
    geometry = pd.DataFrame({"x": x_coords, "y": y_coords})

    # Create atlas (gene expression patterns for training genes)
    atlas_data = np.random.rand(n_spots, len(sample_gene_names["training"]))
    atlas = pd.DataFrame(
        atlas_data, columns=sample_gene_names["training"], index=range(n_spots)
    )

    # Create validation data
    validation_data = np.random.rand(n_spots, len(sample_gene_names["validation"]))
    validation = pd.DataFrame(
        validation_data, columns=sample_gene_names["validation"], index=range(n_spots)
    )

    # Create raw count data (scRNAseq)
    all_genes = sample_gene_names["training"] + sample_gene_names["validation"]
    raw_count_data = np.random.randint(0, 100, size=(n_cells, len(all_genes)))
    raw_count = pd.DataFrame(
        raw_count_data, columns=all_genes, index=[f"cell_{i}" for i in range(n_cells)]
    )

    # Create macrochaete data (morphological marks)
    macrochaete = pd.DataFrame(np.random.randint(0, 20, size=(10, 2)))

    # Ensure geometry has the same index as atlas
    geometry.index = atlas.index

    # Save the data to the tmp_path
    data_path = tmp_path / "data"
    data_path.mkdir()

    atlas_path = data_path / "atlas.csv"
    validation_path = data_path / "validation.csv"
    geometry_path = data_path / "geometry.csv"
    raw_count_path = data_path / "raw_count.csv"
    macrochaete_path = data_path / "macrochaete.csv"

    atlas.to_csv(atlas_path)
    validation.to_csv(validation_path)
    geometry.to_csv(geometry_path)
    raw_count.to_csv(raw_count_path)
    macrochaete.to_csv(macrochaete_path, header=False, index=False)

    return {
        "atlas": atlas,
        "validation": validation,
        "geometry": geometry,
        "raw_count": raw_count,
        "macrochaete": macrochaete,
        "paths": {
            "atlas": atlas_path,
            "validation": validation_path,
            "geometry": geometry_path,
            "raw_count": raw_count_path,
            "macrochaete": macrochaete_path,
        },
    }


@pytest.fixture
def temp_output_path():
    """Create a temporary directory for saving output files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname
