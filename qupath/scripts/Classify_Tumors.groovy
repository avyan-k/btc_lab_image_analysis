// Setting Path for loading extensions
path = System.getProperty("user.home")+"/QuPath/v0.5"
qupath.lib.gui.prefs.PathPrefs.userPathProperty().set(path)
print('\n' +qupath.lib.gui.prefs.PathPrefs.userPath+'\n')
// Loading PyTorch
import ai.djl.engine.Engine
import qupath.ext.djl.DjlTools
if(!DjlTools.isEngineAvailable("PyTorch")){
   // downloads PyTorch Engine by calling DJL extension methods, files will be installed in ~/.djl.ai
    DjlTools.getEngine("PyTorch", true)
}
// verify installation
import ai.djl.pytorch.jni.LibUtils
println "LibTorch: " + LibUtils.getLibTorch().dir
import qupath.ext.wsinfer.WSInfer

//Inference
createAnnotationsFromPixelClassifier(System.getProperty("user.dir")+"/qupath/classifiers/pixel_classifiers/Full Tissue.json", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")
baseThreshold = 0.9
belowBaseThresholdClass = "Other"

selectAnnotations();
WSInfer.runInference("DDC_UC_1-10000-Unnormalized")
def tiles = getTileObjects()
tiles.each { t ->

    t.setColor(ColorTools.packARGB(
    (int) (t.measurements.get("normal")*255.0)+1,
    143,
    28,
    49,
    )) //red
    t.setColor(ColorTools.packARGB(
    (int) (t.measurements.get("well_diff")*255.0)+1,
    153,
    204,
    153,
    )) //green
    t.setColor(ColorTools.packARGB(
    (int) (t.measurements.get("undiff")*255.0)+1,
    77,
    102,
    204,
    )) //blu

//    def maximum = Collections.max(t.measurements.entrySet(), Map.Entry.comparingByValue())
//        if(maximum.getValue() >= baseThreshold) {
//           t.classifications = [maximum.getKey()]
//           if (maximum.getKey().equals("normal")) {
//              t.setColor(143,28,49) //red
//           } else if (maximum.getKey().equals("well_diff")) {
//               t.setColor(153,204,153) //green
//           } else {
//              t.setColor(77,102,204)  //blue
//           }
//        }
//        else {
//           t.classifications = [belowBaseThresholdClass]
//    t.setColor(ColorTools.packARGB(
//    1,
//    0,
//    0,
//    0,
//    )) //red
//)
}
runPlugin('qupath.lib.plugins.objects.TileClassificationsToAnnotationsPlugin', '{"pathClass":"All classes","deleteTiles":false,"clearAnnotations":false,"splitAnnotations":false}')
def classes = tiles[0].measurements.keySet()
for (tumorclass : classes) {
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


// Export image and measurements
def sep = ","
resultsPath = System.getProperty("user.dir")+ System.getProperty("file.separator") + "measurements" + System.getProperty("file.separator")

imageData = getCurrentServer().getMetadata()
imageName = imageData.getName().toString()
imageFolder = resultsPath + imageData.getName() + System.getProperty("file.separator")

def measurementsPath = imageFolder+ "measurements.csv"
def imageDataPath = imageFolder +"image_data.csv"
def imagePath = imageFolder +"resized_image.png"

def imageoutputFile = new File(imageDataPath)
imageoutputFile.createNewFile()

imageoutputFile.text = "Name," + imageName + "\n"

imageoutputFile.append("Image Height,"+imageData.getHeight() + "\n")
imageoutputFile.append("Image Width,"+imageData.getWidth() + "\n")

imageoutputFile.append("Pixel Height,"+imageData.getPixelHeightMicrons() + "\n")
imageoutputFile.append("Pixel Width,"+imageData.getPixelWidthMicrons() + "\n")

imageoutputFile.append("Downsample Level,"+imageData.getLevels()[5].getDownsample() + "\n")
imageoutputFile.append("Pixel Height,"+imageData.getLevels()[5].getHeight() + "\n")
imageoutputFile.append("Pixel Width,"+imageData.getLevels()[5].getWidth())

//print imageData.getHeight()
//print imageData.getWidth()
//print imageData.getPixelHeightMicrons()
//print imageData.getPixelWidthMicrons()
//print imageData.getLevels()[5].getDownsample()
//print imageData.getLevels()[5].getHeight()
//print imageData.getLevels()[5].getWidth()

def outputFile = new File(measurementsPath)
outputFile.createNewFile()
outputFile.text = "Name,x,y,normal,undiff,well_diff,Class"
tiles.each { t ->
   outputFile.append("\n"+t.getName() + sep)
   outputFile.append(t.getROI().getBoundsX() + sep)
   outputFile.append(t.getROI().getBoundsY() + sep)
   for(String key:t.getMeasurements().keySet()) {
      outputFile.append(+t.getMeasurements()[key] + sep)
   }
   outputFile.append(t.getPathClass())
}

def server = getCurrentServer()

def requestFull = RegionRequest.createInstance(server, 32)
writeImageRegion(server, requestFull, imagePath)
print "Done!"