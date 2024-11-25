def sep = ","
resultsPath = getProject().getPath().getParent().getParent().toString()+ System.getProperty("file.separator") + "measurements" + System.getProperty("file.separator")

imageData = getCurrentServer().getMetadata()
imageName = imageData.getName().toString()
imageFolder = resultsPath + imageData.getName() + System.getProperty("file.separator")

def measurementsPath = imageFolder+ "measurements.csv"
def imagePath = imageFolder +"image_data.csv"

selectAnnotations();
def tiles = getTileObjects()

def imageoutputFile = new File(imagePath)
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

//print tiles[0].getName()
//print tiles[0].getROI().getBoundsX()
//print tiles[0].getROI().getBoundsY()
//
//for(String key:tiles[0].getMeasurements().keySet()) {
//   print tiles[0].getMeasurements()[key]
//}
//print tiles[0].getPathClass()


print "Done!"