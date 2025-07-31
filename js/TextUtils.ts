/**
 * Returns the maximum font size that can be drawn within the available width
 * @param availableWidth 
 * @param labels
 * @param svg svg where we are drawing this in.
 * @returns 
 */
export function getMaxFontSize(availableWidth: number, labels: string[], svg: d3.Selection<SVGGElement, unknown, HTMLElement, any>): number {
    let dummyText = svg.append("text")
    let currentFontSize = 900; //if 12 size font fits, use this. Otherwise use a smaller font

    //go through all labels, set font-size to fit.
    for (let label of labels) {
        dummyText.text(label);
        dummyText.node()!.style.fontSize = currentFontSize + "px";
        let textWidth = dummyText.node()!.getBBox().width;
        //calculate how far we are off.
        let percentage = availableWidth / textWidth;
        if (percentage < 1) { //if it doesn't fit, reduce the font size by the required percentage.
            currentFontSize *= percentage;
        }
    }
    dummyText.remove();
    return currentFontSize;
}

export function getTextWidth(svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>, fontSize: string, text: string) {
    const tempText = svg.append("text").text(text).attr("font-size", fontSize).node()!;
    const textWidth = tempText.getBBox().width;
    tempText.remove(); // Clean up the temp text
    return textWidth;
}

export function trimText(text: string, length: number): string {
    if (text.length <= length) {
        return text;
    }
    return text.slice(0, length - 3) + "...";
}

export function trimTextWidth(svg: d3.Selection<SVGSVGElement, unknown, HTMLElement, any>, text: string, fontSize: string, maxWidth: number): string {
    let textWidth = getTextWidth(svg, fontSize, text)

    // If the text width is within the maxWidth, return the original text
    if (textWidth <= maxWidth) {
        return text;
    }

    // Trim the text until it fits within the maxWidth
    let trimmedText = text;
    while (textWidth > maxWidth && trimmedText.length > 0) {
        trimmedText = trimmedText.slice(0, -1);
        textWidth = getTextWidth(svg, fontSize, trimmedText + "...");
    }

    return trimmedText + "...";
}