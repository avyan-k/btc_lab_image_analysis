String model = "local/DDC_UC_1-Unnormalized/torchscript_model.pt";
createAnnotationsFromPixelClassifier("Full Tissue", 150000.0, 0.0, "SPLIT", "DELETE_EXISTING", "INCLUDE_IGNORED")
selectAnnotations();