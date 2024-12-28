import qupath.lib.images.servers.ImageServers
import java.io.File
// def project = getProject()
// if (project == null) {
//     println "A project must be opened before running this script"
//     return
// }



File folder = new File(args[0].toString());
for (File imagePath : folder.listFiles()) {
        // def imageServer = ImageServers.buildServer(imagePath)
        // describe(imageServer)
        System.out.println(imagePath.getName());
}