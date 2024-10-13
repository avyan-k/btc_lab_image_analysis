createAnnotationsFromPixelClassifier("Full Tissue", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")

baseThreshold = 0.9
belowBaseThresholdClass = "Other"

selectAnnotations();
qupath.ext.wsinfer.WSInfer.runInference("DDC_UC_1-Unnormalized")
def tiles = getTileObjects()
tiles.each { t ->
    def maximum = Collections.max(t.measurements.entrySet(), Map.Entry.comparingByValue())
    if(maximum.getValue() >= baseThreshold) {
       t.classifications = [maximum.getKey()] 
    }
    else {
       t.classifications = [belowBaseThresholdClass] 
    }
}
runPlugin('qupath.lib.plugins.objects.TileClassificationsToAnnotationsPlugin', '{"pathClass":"All classes","deleteTiles":false,"clearAnnotations":false,"splitAnnotations":false}')
def classes = tiles[0].measurements.keySet()
for (tumorclass:classes) {
    selectObjectsByClassification(tumorclass);
    mergeSelectedAnnotations();
    resetSelection();
}
selectObjectsByClassification("Other");
mergeSelectedAnnotations()
resetSelection()
selectObjectsByClassification("Region*");
mergeSelectedAnnotations()
resetSelection()