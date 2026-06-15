//setBatchMode(true);
x = 0; //the x-start coordinate
y = 0; //the y-start coordinate
width = 533.333333333; //the width of the rectangle
height = 600; //the height of the rectangle
spacing = 0; //spacing between the rectangles
numRow = 2; //how many rows
numCol = 3; //how many columns


/*Create the selections and add them to the ROI Manager!*/
for (i = 0; i < numRow; i++) {
    for (j = 0; j < numCol; j++) {
        xOffset = j * (width + spacing);
        yOffset = i * (height + spacing);
        /*Create a rectangular selection!*/
        makeRectangle(x + xOffset, y + yOffset, width, height);
        /*Add the selection to the ROI Manager!*/
        roiManager("Add");
    }
}
/*Show all selections in the image with (mouse) selectable labels!*/
roiManager("Show All with labels");
mainTitle=getTitle();
for (u = 0; u < roiManager("count"); ++u) {
    run("Duplicate...", "title=crop duplicate");
    roiManager("Select", u);
    run("Crop");
    saveAs("Tiff", "C:\\Users\\Lab-Guest\\Desktop\\Brian\\20250526 BCell ML\\B Cell + M0s\\30 um\\R2\\" + "Grid_Section_" + (u + 1) + ".tif");
    close();
    //Next round!
    selectWindow(mainTitle);
}