import ai.djl.engine.Engine
import qupath.ext.djl.DjlTools
import ai.djl.util.Utils
import ai.djl.pytorch.jni.LibUtils
import ai.djl.util.cuda.CudaUtils
import qupath.ext.wsinfer.WSInfer

String fs = System.getProperty("file.separator")
// Setting Path for loading extensions

// check if script is run from project
if (getProject() != null) {
    resultsPath = String.join(fs,getProject().getPath().getParent().getParent().toString(), "results","inference") 
    classifierPath = "Full Tissue"
}
else {
   resultsPath = String.join(fs,System.getProperty("user.dir"), "results","inference") 
   classifierPath = String.join(fs,System.getProperty("user.dir").toString(),"qupath","classifiers","pixel_classifiers","Full Tissue.json")
}
println(resultsPath)

// verify installation
import ai.djl.pytorch.jni.LibUtils
println "Pytorch Library Path: " + LibUtils.getLibTorch().dir
println("GPU count: "+CudaUtils.getGpuCount())


// Inference
Downsample = 2
baseThreshold = 0.9
belowBaseThresholdClass = "Other"
//clearAllObjects();
//createAnnotationsFromPixelClassifier(classifierPath, 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")
//selectAnnotations();
//runPlugin('qupath.lib.algorithms.TilerPlugin', '{"tileSizeMicrons":117.6064,"trimToROI":true,"makeAnnotations":false,"removeParentAnnotation":false}')
//
//
//
//selectTiles();
//import qupath.ext.wsinfer.ui.WSInferPrefs
//// Set parallel tile loaders
//WSInferPrefs.numWorkersProperty().setValue(16);
//// Set batch size
//WSInferPrefs.batchSizeProperty().setValue(64);
//println(WSInferPrefs.batchSizeProperty().getValue())
//qupath.ext.wsinfer.WSInfer.runInference("DDC_UC_1-10000-Unnormalized")

def tiles = getTileObjects()
tiles.each { t ->
    def maximum = Collections.max(t.measurements.entrySet(), Map.Entry.comparingByValue())
    if(maximum.getValue() >= baseThreshold) {
        t.classifications = [maximum.getKey()]
        if (maximum.getKey().equals("normal")) {
            t.setColor(ColorTools.packARGB(
            (int) (t.measurements.get("normal")*255.0)+1,
            143,
            28,
            49,
            )) //red          
        } 
        else if (maximum.getKey().equals("well_diff")) {
            t.setColor(ColorTools.packARGB(
            (int) (t.measurements.get("well_diff")*255.0)+1,
            153,
            204,
            153,
            )) //green
        } 
        else {
            t.setColor(ColorTools.packARGB(
            (int) (t.measurements.get("undiff")*255.0)+1,
            77,
            102,
            204,
            )) //blue
        }
    }
    else {
        t.classifications = [belowBaseThresholdClass]
        t.setColor(ColorTools.packARGB(
        255,
        255,
        200,
        0
        )) //yellow
    }
}
String[] classes = tiles[0].measurements.keySet()
//runPlugin('qupath.lib.plugins.objects.TileClassificationsToAnnotationsPlugin', '{"pathClass":"All classes","deleteTiles":false,"clearAnnotations":false,"splitAnnotations":false}')
//for (tumorclass : classes) {
//    selectObjectsByClassification(tumorclass);
//    mergeSelectedAnnotations();
//    resetSelection();
//}
//selectObjectsByClassification("Other");
//mergeSelectedAnnotations()
//resetSelection()
//selectObjectsByClassification("Region*");
//mergeSelectedAnnotations()
//resetSelection()

// Export image and measurements
String sep = ","

def imageData = getCurrentServer().getMetadata()
String imageName = imageData.getName().toString()

String imageFolder = String.join(fs,resultsPath,imageName.substring(0, imageName.lastIndexOf('.')),"measurements") + fs
File imageDirectory = new File(imageFolder);
if (!imageDirectory.exists()){
    imageDirectory.mkdirs();
}

String measurementsPath = imageFolder+ "measurements.csv"
String imageDataPath = imageFolder +"image_data.csv"
String imagePath = imageFolder +"resized_image.png"
// Save WSI metadata
File imageoutputFile = new File(imageDataPath)
imageoutputFile.createNewFile()

imageoutputFile.text = "Name," + imageName + "\n"

imageoutputFile.append("Pixel Height,"+imageData.getPixelHeightMicrons() + "\n")
imageoutputFile.append("Pixel Width,"+imageData.getPixelWidthMicrons() + "\n")

imageoutputFile.append("Image Height,"+imageData.getHeight() + "\n")
imageoutputFile.append("Image Width,"+imageData.getWidth() + "\n")

imageoutputFile.append("Tile Height," + tiles[0].getROI().getBoundsWidth() + "\n")
imageoutputFile.append("Tile Width," + tiles[0].getROI().getBoundsHeight() + "\n")

imageoutputFile.append("Downsample Level,"+imageData.getLevels()[Downsample].getDownsample() + "\n")
imageoutputFile.append("Downsample Height,"+imageData.getLevels()[Downsample].getHeight() + "\n")
imageoutputFile.append("Downsample Width,"+imageData.getLevels()[Downsample].getWidth()+ "\n")

imageoutputFile.append("Tumor Classes"+sep+classes[0])
for(i=1;i<classes.length;i++) {
    imageoutputFile.append(sep+classes[i])
}
// Save tile measurements
File outputFile = new File(measurementsPath)
outputFile.createNewFile()

outputFile.text = "Name,x,y," + String.join(sep,classes) + ",Class"

tiles.each { t ->
   outputFile.append("\n"+t.getName() + sep)
   outputFile.append(t.getROI().getBoundsX() + sep)
   outputFile.append(t.getROI().getBoundsY() + sep)
   for(tumorclass : classes) {
       outputFile.append(t.getMeasurements()[tumorclass] + sep)
   }
   outputFile.append(t.getPathClass())
}
// Save downscaled WSI
def server = getCurrentServer()

def requestFull = RegionRequest.createInstance(server, 2**Downsample)
writeImageRegion(server, requestFull, imagePath)
print "Done!"