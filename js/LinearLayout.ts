import * as PIXI from 'pixi.js';

/**
 * Horizontal margin between containers
 */
const horizontalMargin = 10;

/**
 * vertical margin between containers
 */
const verticalMargin = 15;

/**
 * (Re)positions all containers such that they fit the available width.
 * @returns the maximal y2 coordinate.
 * */
export function setContainerPositions(containerList: PIXI.Container[], maxWidth: number): number {

    //set position of containers.
    let currentX = 0;//Current x-position we intend to place the next container.
    let currentY = 0//current y-position we intend to place the next container

    let maxHeightInRow = 0;//Holds the maximum height of a container in the current row
    for (let container of containerList) {
        let newWidth = container.getBounds().width;
        if ((currentX + newWidth + horizontalMargin) > maxWidth) { //Doesn't fit anymore, start new horizontal row
            if (currentX != 0) { //if the first element in a row doesn't fit, place it anyway and don't start a new row.
                currentX = 0;
                currentY += container.getBounds().height + verticalMargin;
                maxHeightInRow = 0
            }
        }
        container.x = currentX
        container.y = currentY;
        currentX += newWidth + horizontalMargin;
        maxHeightInRow = Math.max(maxHeightInRow, container.getBounds().height)
    }

    //return the maximum y2 coordinate
    return currentY + maxHeightInRow;
}


/**
 * (Re)positions all containers such that they fit the available width and height.
 * Assumes all containers have the same height and width
 * @returns the maximal y2 coordinate.
 * */
export function setContainerPositionsAndScale(containerList: PIXI.Container[], maxWidth: number, maxHeight: number) {

    let currentContainerWidth: number;
    let currentContainerHeight: number;
    let containerCount = 0;
    for (let container of containerList) {
        container.x = 0;//remove previous scaling and transformations
        container.y = 0;
        container.scale.x = 1;
        container.scale.y = 1;
        currentContainerWidth = container.getBounds().width; //All the same.
        currentContainerHeight = container.getBounds().height;//All the same.
        containerCount++;
    }

    let scalingFactor = calculateScalingFactor(containerCount, currentContainerWidth!, currentContainerHeight!, maxWidth, maxHeight);
    //console.log(scalingFactor)

    for (let container of containerList) {
        container.scale.x = scalingFactor;
        container.scale.y = scalingFactor;
    }
    setContainerPositions(containerList, maxWidth);
}

/**
 * Calculates the maximum scaling factor such that all containers fit in the view
 * @param containerCount
 * @param width 
 * @param height 
 * @param maxWidth 
 * @param maxHeight 
 * @return the maximum scaling factor
 */
function calculateScalingFactor(containerCount: number, width: number, height: number, maxWidth: number, maxHeight: number): number {
    maxWidth = maxWidth;
    maxHeight = maxHeight;

    let maxScaleFactor = 0;
    //Go through all possibilities of containers per row, and pick the one with the highest scale factor
    for (let containersPerRow = containerCount; containersPerRow > 0; containersPerRow--) {
        let scaleX = maxWidth / ((width+horizontalMargin) * containersPerRow)
        let rows = Math.ceil(containerCount / containersPerRow);
        let scaleY = maxHeight / ((height+verticalMargin) * rows); 
        let scaleFactor = Math.min(scaleX, scaleY);
        maxScaleFactor = Math.max(scaleFactor, maxScaleFactor);
    }
    //round it to prevent some jiggling
    maxScaleFactor = Math.floor(maxScaleFactor*1000)/1000;
    return maxScaleFactor;
}


