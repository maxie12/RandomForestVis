import { getCurrentMainPlotWidth, viewPort } from '../index';
import * as PIXI from 'pixi.js';

//holds the tooltip. We move this to position is correctly
let toolTipContainer: PIXI.Container;
let toolTipBackground: PIXI.Graphics;
let toolTipText: PIXI.BitmapText;

const verticalOffset = 3;
const horizontalOffset = 6;
const baseFontSize = 24;

const style = new PIXI.TextStyle({
    fill: "black",
    textBaseline: "bottom",
    fontSize: baseFontSize
});

export function initializeBarCodeToolTip(container: PIXI.Container): PIXI.Container {

    toolTipContainer = new PIXI.Container();
    toolTipBackground = new PIXI.Graphics();
    toolTipContainer.addChild(toolTipBackground);


    // toolTipText = new PIXI.Text({ text: "", style }); //initialize blank
    toolTipText = new PIXI.BitmapText({ text: "", style }); //initialize blank
    toolTipText.x = horizontalOffset; //offset the text to be positioned nicely.
    toolTipText.y = verticalOffset;
    toolTipContainer.addChild(toolTipText);

    container.addChild(toolTipContainer);
    return toolTipContainer;
}

/**
 * Sets the barcode to the right position and text. Does NOT call the rendering function
 * @param x 
 * @param y 
 * @param text 
 */
export function showBarCodeToolTip(x: number, y: number, text: string) {
    let scale = viewPort.worldWidth / viewPort.width;
    let offSetX = horizontalOffset * scale
    let offSetY = verticalOffset * scale


    let targetFontSize = baseFontSize * scale;
    style.fontSize = targetFontSize;
    let newToolTipWidth = PIXI.CanvasTextMetrics.measureText(text, style).width + offSetX * 2;
    let newToolTipHeight = (targetFontSize + offSetY * 2) * 2; //we have 1 line break so * 2

    let tooltipPosition = viewPort.toWorld(x, y);
    // let tooltipPosition = viewPort.toWorld(x + horizontalOffset, y - verticalOffset);
    let tooltipX = tooltipPosition.x + offSetX;
    let tooltipY = tooltipPosition.y - newToolTipHeight - offSetY;


    //make sure we don't go out of bounds (technically can still go out of bounds on left side)
    if (tooltipX + newToolTipWidth > (viewPort.left + viewPort.screenWidthInWorldPixels)) {
        tooltipX = tooltipPosition.x - offSetX - newToolTipWidth;
    }


    toolTipContainer.x = tooltipX;
    toolTipContainer.y = tooltipY;


    //set the size
    let upscaleFontSize = 90;
    toolTipText.style.fontSize = upscaleFontSize;
    toolTipText.scale.set(targetFontSize / upscaleFontSize) //need to do scaling to ensure that the text is legible and not blurred.

    toolTipText.text = text;
    toolTipText.x = offSetX;
    toolTipText.y = offSetY * 2;

    //delete the previous rectangle, draw a new rounded rectangle with the correct width
    toolTipBackground
        .clear()
        .roundRect(0, 0, newToolTipWidth, newToolTipHeight, newToolTipHeight / 5)
        .fill("white")
        .stroke({ width: scale, color: "black" })

    toolTipContainer.visible = true;
}


/**
 * Hides the barcodetooltip. Does NOT call the rendering function
 */
export function hideBarCodeToolTip() {
    toolTipContainer.visible = false;
}