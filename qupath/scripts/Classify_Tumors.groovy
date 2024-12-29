// Setting Path for loading extensions
path = System.getProperty("user.home")+"/QuPath/v0.5"
qupath.lib.gui.prefs.PathPrefs.userPathProperty().set(path)
print('\n' +qupath.lib.gui.prefs.PathPrefs.userPath+'\n')
// Loading PyTorch
import ai.djl.engine.Engine
import qupath.ext.djl.DjlTools
if(!DjlTools.isEngineAvailable("PyTorch")){
   // downloads PyTorch Engine
    DjlTools.getEngine("PyTorch", true)
}
// verify installation
import ai.djl.pytorch.jni.LibUtils
println "LibTorch: " + LibUtils.getLibTorch().dir
import qupath.ext.wsinfer.WSInfer

createAnnotationsFromPixelClassifier(System.getProperty("user.dir")+"/classifiers/pixel_classifiers/Full Tissue.json", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")
// CHANGE ABOVE TO BELOW IF RUNNING DIRECTLY FROM QUPATH APPLICATION
createAnnotationsFromPixelClassifier("Full Tissue", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")
baseThreshold = 0.9
belowBaseThresholdClass = "Other"

selectAnnotations();
WSInfer.runInference("DDC_UC_1-10000-Unnormalized")
def tiles = getTileObjects()
tiles.each { t ->
    def maximum = Collections.max(t.measurements.entrySet(), Map.Entry.comparingByValue())
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
