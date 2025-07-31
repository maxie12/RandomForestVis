import * as d3 from "d3";
import { Classifications, Predictions } from "../types";
import { trimTextWidth } from "../TextUtils";
import { classHovered, classStopHover, initClassInteraction, initInteractionClasses, toggleClassificationSelected, toggleOriginalClassSelected } from "../Interactions";

let classificationRectangles: d3.Selection<SVGRectElement, unknown, HTMLElement, any>[][] = [];
let classRectangles: d3.Selection<SVGRectElement, unknown, HTMLElement, any>[] = [];

let allPredictions: Predictions;

export function createClassChartMatrix(target: string, classes: { [className: string]: { count: number, index: number } }, colors: (t: number) => string, classification: Classifications) {  

    let classCount = Object.entries(classes).length;
    allPredictions = getAllPredictions(classification, classCount);

    //Let the interactions know which classes are present.
    initInteractionClasses(classCount, allPredictions);

    let maxPredictionsInClass = 0; //holds the maximumamount of predictions in a single class, used for scaling
    for (let classI = 0; classI < classCount; classI++) {
        maxPredictionsInClass = Math.max(maxPredictionsInClass, allPredictions[classI].count)
    }

    let tooltip = d3.select(target)
        .append("div")
        .attr("id", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden");


    d3.select(target).selectAll('*').remove();//@ts-ignore
    let width = d3.select(target).node().getBoundingClientRect().width;
    let heightPerClass = 20;
    let horpadding = 2;
    let rectangleWidth = 20;
    let verticalMargin = 5;
    let height = (classCount * heightPerClass) + verticalMargin;

    let svg = d3.select(target)
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    //initialize the interaction for the classes
    initClassInteraction(classCount);

    let y = 0;
    for (let classI = 0; classI < classCount; classI++) {
        classificationRectangles[classI] = [];
        let className = getClassName(classI, classes);

        //get classifications
        let classificationArray: number[] = [];
        let total = 0;
        for (let class2I = 0; class2I < classCount; class2I++) {
            classificationArray[class2I] = allPredictions[classI][class2I];
            total += allPredictions[classI][class2I];
        }


        let g = svg.append("g");
        //draw initial rectangle 
        let classRectangle = g.append("rect")
            .attr("x", 0)
            .attr("y", y)
            .attr("class", "bar")
            .attr("width", rectangleWidth)
            .attr("height", heightPerClass)
            .attr("fill", colors(classI / classCount))
            .style("cursor", "pointer")


        let correctClassificationsCount = allPredictions[classI][classI];
        let percentage = Math.floor(correctClassificationsCount / total * 100);
        classRectangle.append("title").text(`${className} is classified correctly ${correctClassificationsCount} out of ${total} times (${percentage}%) in the test data.`);


        classRectangles[classI] = classRectangle;

        let maxFontWidth = 85; //Magic number. Determined by a string of "W"'s using the specified font size.
        //TODO: Automatically detect how much width is needed, and use that. Can calculate if the names are known

        //draw label after the rectangle
        g.append("text")
            .attr("x", rectangleWidth + horpadding)
            .attr("y", y + 15) //15 is equal to the fontsize
            .attr("font-size", "15px")
            .attr("class", "bar-text")
            .text(function () {
                return trimTextWidth(svg, className, "15px", maxFontWidth);
            })
            .append("title").text(className);


           g.on("mouseover",() => classHovered(classI))
            .on("mouseleave",() => classStopHover(classI))
            .on("click", () => toggleOriginalClassSelected(classI))
        //TODO: Automatically detect how much width is needed, and use that. Can calculate if the names are known


        //draw the row from the confusionmatrix
        const startX = rectangleWidth + horpadding + maxFontWidth;
        const percentageFromMax = total / maxPredictionsInClass;
        const x = d3.scaleLinear()
            .domain([0, total])
            .range([0, (width - startX) * percentageFromMax]);

        let dataG = svg.append("g")

        let xPos = startX;
        for (let class2I = 0; class2I < classCount; class2I++) {
            let className2 = getClassName(class2I, classes);

            //no need to draw bars that don't exist.
            let barWidth = x(classificationArray[class2I])
            let rect = dataG.append("rect")
                .attr("class", "bar")
                .attr("x", xPos)
                .attr("y", y)
                .attr("height", heightPerClass)
                .attr("width", barWidth)
                .attr("fill", colors(class2I / classCount))
                .style("cursor", "pointer")
                .on("mouseover",() => classHovered(classI,class2I))
                .on("mouseleave",() => classStopHover(classI,class2I))
                .on("click", function () {
                    toggleClassificationSelected(classI, class2I)
                });

            //round to full number. Use floor to not overestimate
            let percentageClass2 = Math.floor(classificationArray[class2I] / total * 100);

            rect.append("title").text(`${className} was classified as ${className2} by the model ${classificationArray[class2I]} out of ${correctClassificationsCount} times (${percentageClass2}%) in the test data.`)
            classificationRectangles[classI][class2I] = rect;
            xPos += barWidth;

        }
        y += heightPerClass;
    }
    d3.select("#model-accuracy").text("Test accuracy: " + classification.accuracy.toFixed(2).toString())
}

export function dimClasses(shownClasses: boolean[][]) {
    let classCount = shownClasses.length;
    for (let i = 0; i < classCount; i++) {
        for (let j = 0; j < classCount; j++) {
            let classificationRect = classificationRectangles[i][j];
            if (shownClasses[i][j]) {
                classificationRect.node()!.classList.remove("classDeselected");
            } else {
                classificationRect.node()!.classList.add("classDeselected");
            }
        }
    }
}

function getAllPredictions(classification: Classifications, classCount: number) {

    let predictions: Predictions = { count: 0 };

    //initialize matrix
    for (let originalClassI = 0; originalClassI < classCount; originalClassI++) {
        predictions[originalClassI] = { count: 0 }
        for (let targetClassI = 0; targetClassI < classCount; targetClassI++) {
            predictions[originalClassI][targetClassI] = 0;
        }
    }
    //add the correct predictions
    for (let originalClassI = 0; originalClassI < classCount; originalClassI++) {
        if (classification.correctPredictions[originalClassI] !== undefined) {
            predictions[originalClassI][originalClassI] = classification.correctPredictions[originalClassI];
            predictions[originalClassI].count = classification.correctPredictions[originalClassI]
            predictions.count += classification.correctPredictions[originalClassI]
        }
    }

    //add the incorrect predictions
    for (let originalClassI = 0; originalClassI < classCount; originalClassI++) {
        if (classification.wrongPredictions[originalClassI] == undefined) {
            continue;
        }
        for (let targetClassI = 0; targetClassI < classCount; targetClassI++) {
            if (classification.wrongPredictions[originalClassI][targetClassI] == undefined) {
                continue;
            }
            predictions[originalClassI][targetClassI] = classification.wrongPredictions[originalClassI][targetClassI];
            predictions[originalClassI].count += classification.wrongPredictions[originalClassI][targetClassI];
            predictions.count += classification.wrongPredictions[originalClassI][targetClassI]
        }
    }
    return predictions;
}

function getClassName(index: number, classes: { [className: string]: { count: number, index: number } }): string {
    for (let className in classes) {
        if (classes[className].index == index) {
            return className;
        }
    }
    return "ClassNotFound";
}

function allUsedClassesAreDeSelected(array: boolean[], classIndex: number) {
    for (let j = 0; j < array.length; j++) {
        if (allPredictions[classIndex][j] == 0) {//ignore unused classes.
            continue;
        }
        //if a used class is selected, not everything is deselected
        if (array[j] == true) {
            return false;
        }
    }
    return true;
}

function originalClassHovered(classI: number, arg1: boolean): void {
    throw new Error("Function not implemented.");
}
