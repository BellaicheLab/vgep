def test_evaluation_public_surface():
    import vgep
    assert hasattr(vgep, "evaluation")

    from vgep import evaluation
    for name in (
        "per_gene_pearson_r",
        "plot_cdf_curves", "plot_survival_curves", "plot_protein_vs_fish_cdf",
    ):
        assert hasattr(evaluation, name), f"evaluation missing {name!r}"
