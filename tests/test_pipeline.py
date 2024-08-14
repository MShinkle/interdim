from interdim.pipeline import InterDimAnalysis

def test_interdim_analysis_pipeline(small_dataset):
    analysis = InterDimAnalysis(small_dataset, verbose=False)
    
    # Test reduction
    reduced = analysis.reduce(method='pca', n_components=2)
    assert reduced.shape == (100, 2)
    
    # Test clustering
    labels = analysis.cluster(method='kmeans', n_clusters=3)
    assert len(labels) == 100
    
    # Test scoring
    score = analysis.score(method='silhouette')
    assert -1 <= score <= 1