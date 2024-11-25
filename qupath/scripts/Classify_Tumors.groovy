//createAnnotationsFromPixelClassifier("Full Tissue", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")

baseThreshold = 0.9
belowBaseThresholdClass = "Other"

selectAnnotations();
//qupath.ext.wsinfer.WSInfer.runInference("DDC_UC_1-10000-Unnormalized")
def tiles = getTileObjects()
tiles.each { t ->
    def maximum = Collections.max(t.measurements.entrySet(), Map.Entry.comparingByValue())
        print(t.measurements)
        if(maximum.getValue() >= baseThreshold) {
           t.classifications = [maximum.getKey()]
           if (maximum.getKey().equals("normal")) {
              t.setColor(143,28,49) //red
           } else if (maximum.getKey().equals("well_diff")) {
               t.setColor(153,204,153) //green
           } else {
              t.setColor(77,102,204)  //blue
           }
        }
        else {
           t.classifications = [belowBaseThresholdClass]
           t.setColor(255,200,0) //yellow
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