import { Cluster, LeafNode, Tree, Predictions, DimRectangle } from "../types";
import { get1dOrder } from "./Orderer";
import { getDistanceMatrixFromLeafs } from "../Metrics/TreeDistance";
import { getFeatureImportance } from "../Metrics/FeatureImportance";
import { getFeatureCount, mapLeafNodes } from "./Tree";
import * as PIXI from 'pixi.js';
import { selectCluster, focusFeature, unfocusFeature, classHovered, classStopHover } from "../Interactions";
import { classColors, featureColors } from '../colors';
import { selectClusterProjection } from "../Projection";
import { printDuration, printStartTime } from "../Timing";
import { hideBarCodeToolTip, showBarCodeToolTip } from "./Tooltip";


/**
 * Height of the full range of a bar
 */
const verScale = 80;

/**
 * Width of the smallest scale bar chart
 */
const horScale = 5;

const lineWidth = 2;

/**
 * extra space between bars in pixels
 */
const extraHorSpaceBetweenBars = 0;

/**
 * extra space between bars in pixels
 */
const extraHorSpaceBetweenClasses = 2;

/**
 * extra space between features in pixels
 */
const extraVerSpaceBetweenFeatures = lineWidth * 2;


/**
 * Holds the y startCoordinate of each feature in the dataset as well as endcoordinate for the last feature. Takes into account irregular heights
 */
let featureHeight: number[] = [];

/**
 * Holds the list of classes
 */
let classes: string[];

/**
 * Holds groups of dimRuleRectangles. Each group belongs to the same vertical column
 */
let dimRuleRectanglesGroups: DimRectangle[][] = [];

//Holds everything for drawing the horizontal bars and highlighting them
let boundaryCoordinatesPerFeatureIndex: { x1: number, x2: number, y1: number, y2: number }[][];
let boundaryLinePerFeatureIndex: PIXI.Graphics[][];
let boundaryContainers: PIXI.Container[];
let lineColor = new PIXI.Color(`rgb(120,120,120)`);

/**
 * Generated the containers for each barcode. If Cluster is null, no clustering will be applied
 * @param trees 
 * @param cluster 
 * @returns 
 */
export function generateBarcodesForTrees(trees: Tree[], cluster: Cluster, classData: { [className: string]: { count: number, index: number } }): Map<number, PIXI.Container> {
    //use the same order as the indices
    classes = Object.keys(classData);
    classes.sort((a, b) => classData[a].index - classData[b].index);


    // empty for new dataset
    dimRuleRectanglesGroups = [];
    dimRectanglePerFeaturesTop = [[]];
    dimRectanglePerFeaturesBottom = [[]];
    boundaryCoordinatesPerFeatureIndex = [];
    boundaryLinePerFeatureIndex = [];
    boundaryContainers = [];

    setAllPredictions(trees); //For each leafnode, sets the complete matrix of predictions. Do this before merging so we have it for all leaf nodes.
    //TODO: Ensure that the trees are not changing when merging and sorting them. Ideally make a copy at the start.
    if (cluster != null) {
        trees = mergeTrees(trees, cluster);
    }

    setRepresentativePredictions(trees); //Then go into the representatives, and add those. Needs to be done after merging
    trees = sortData(trees);

    setFeatureHeight(trees);
    let containerMapping = new Map();
    //generate a container holding the barcode of each
    trees.forEach((tree) => {
        let container = generateBarcode(tree);
        containerMapping.set(tree.id, container);
    })

    return containerMapping;
}


function getMaxPredictions(tree: Tree): number {
    let maxPredictions = 0;

    tree.leafNodes.forEach((leafNode, i) => {
        let count = 0;
        let allLeafs: LeafNode[] = [];
        if (leafNode.representedLeafNodes != undefined) {
            allLeafs = [...leafNode.representedLeafNodes];
        }
        allLeafs.push(leafNode);
        for (let leaf of allLeafs) {
            if (leaf.wrongPredictions == null) continue;
            count += leaf.correctPredictions + leaf.wrongPredictions.count;
        }
        maxPredictions = Math.max(maxPredictions, count);
    })
    return maxPredictions
}

/**
 * Assumes the data has already been merged
 * @param trees 
 * @returns 
 */
function sortData(trees: Tree[]): Tree[] {
    let order = getFeatureOrder(trees);

    //sort feature and leafnodes.
    for (let tree of trees) {
        //first sort features
        for (let leafNode of tree.leafNodes) {
            leafNode.normalizedRangesPerFeature = sortFeatures(order, leafNode.normalizedRangesPerFeature);
        }
        //then sort based on the ranges of the leafnodes, making sure different classes are split
        sortLeafNodes(tree);
    }

    trees = sortTrees(trees);
    return trees;
}

/**
 * Reorders the ranges within the {normalizedRanges} to have the order specified in {order}
 * @param featureOrder 
 * @param normalizedRanges 
 */
function sortFeatures(featureOrder: number[], normalizedRanges: [number, number][]): [number, number][] {
    //create array to hold the shift
    let updatedNormalizedRanges: [number, number][] = [];
    for (let i = 0; i < normalizedRanges.length; i++) {
        //inset the value at the correct spot
        updatedNormalizedRanges[featureOrder[i]] = normalizedRanges[i];
    }
    return updatedNormalizedRanges;
}

/**
 * Sorts the features based on how much they agree with their merge
 * @param trees 
 * @returns 
 */
function getFeatureOrder(trees: Tree[]): number[] {
    let order: number[] = [];
    for (let i = 0; i < getFeatureCount(trees); i++) {
        order[i] = i;
    }
    //Holds the important per feature. Higher scores are more important.
    let importanceScoreByFeature: number[];
    importanceScoreByFeature = getFeatureImportance(trees, "mergeAgreement");

    //sort on decreasing importance values
    order.sort((a, b) => {
        return importanceScoreByFeature[a] - importanceScoreByFeature[b];
    })

    return order;
}


function sortLeafNodes(tree: Tree) {
    //TODO: Potentially other distance measure


    let currentLeafOrder = tree.leafNodes;

    //sort each class seperatly using dimensionality reduction
    let newLeafNodeOrder: LeafNode[] = [];
    for (let className of classes) {
        let leafNodesOfClass = currentLeafOrder.filter(a => a.class == className);

        let distanceMatrix = getDistanceMatrixFromLeafs(leafNodesOfClass);
        //get 1d dimensionality reduction from distances, and update the order.
        let order = get1dOrder(distanceMatrix);
        let newLeafNodeOrdering = getNewLeafNodeOrder(leafNodesOfClass, order);

        //add the nodes for this class
        newLeafNodeOrder = newLeafNodeOrder.concat(newLeafNodeOrdering);
    }


    // //sort each class based on how frequent the rule is.
    // let predictionCount = new Map<LeafNode, number>();
    // for (let leafNode of currentLeafOrder) {
    //     let predictions = getAllPredictions(leafNode, true);
    //     predictionCount.set(leafNode, predictions.count);
    // }

    // let newLeafNodeOrder: LeafNode[] = [];
    // for (let className of classNames) {
    //     let leafNodesOfClass = currentLeafOrder.filter(a => a.class == className);

    //     leafNodesOfClass.sort((a, b) => (predictionCount.get(b)! - predictionCount.get(a)!))
    //     newLeafNodeOrder = newLeafNodeOrder.concat(leafNodesOfClass)

    // }
    //update it in the tree.
    tree.leafNodes = newLeafNodeOrder;
}

function getNewLeafNodeOrder(leafNodesOfClass: LeafNode[], order: number[]): LeafNode[] {
    let newLeafNodes = [];
    for (let i = 0; i < order.length; i++) {
        let leafNodeAtPosI = leafNodesOfClass[order[i]];
        newLeafNodes[i] = leafNodeAtPosI;

        if (leafNodeAtPosI == undefined) {
            console.error("no such node. position " + order[i] + " does not exist. i=" + i + "max index equals" + leafNodesOfClass.length + "orderlength is " + order.length);
        }
    }
    return newLeafNodes;
}

/**
 * Merges trees according to the cluster. One representative will be picked for each tree.
 * @param trees 
 * @param linkageData
 */
function mergeTrees(trees: Tree[], cluster: Cluster): Tree[] {
    let repMapping = getTreesToMerge(cluster)

    let idMapping = new Map<number, Tree>();
    for (let tree of trees) {
        idMapping.set(tree.id, tree);
    }

    let representedTrees: Tree[] = [];
    for (let repId of repMapping.keys()) {
        let representativeTree = idMapping.get(repId);
        if (representativeTree === undefined) {
            throw new Error(`Tree with id ${repId} is not defined.`);
        }
        for (let representedId of repMapping.get(repId)!) {
            let representedTree = idMapping.get(representedId);
            if (representedTree === undefined) {
                throw new Error(`Tree with id ${representedId} is not defined. Was represnted by tree with id ${repId}`);
            }
            //map the leafnodes of the represented tree to the representative
            mapLeafNodes(representativeTree, representedTree);
        }

        representedTrees.push(representativeTree);
    }
    return representedTrees;
}

/**
 * Holds a mapping from the cluster center to all trees it reresents {CentroidID,[representedIds]}
 * @param cluster 
 */
function getTreesToMerge(cluster: Cluster) {
    let mapping = new Map<number, number[]>();
    for (let [key, centroidId] of cluster.centroids) {

        let representedByCentroid: number[] = [];
        for (let i = 0; i < cluster.labels.length; i++) {
            if (cluster.labels[i] === +key) {
                representedByCentroid.push(i);
            }
        }
        mapping.set(centroidId, representedByCentroid);
    }
    return mapping;
}

/**
 * Sorts the trees based on the distance measure included. Uses a dimensionality reduction technique to 1d.
 * @param trees 
 * @returns 
 */
function sortTrees(trees: Tree[]): Tree[] {

    let distances: number[][] = [];
    for (let i = 0; i < trees.length; i++) {
        distances[i] = trees[i].clusterDistances;
    }

    //get 1d dimensionality reduction
    let order = get1dOrder(distances);

    trees.sort((a, b) => {
        return order.indexOf(a.id) - order.indexOf(b.id);
    })

    return trees;
}


/**
 * Generates a barcode for a single tree, positioned at 0,0 
 * @param tree 
 * @returns 
 */
function generateBarcode(tree: Tree): PIXI.Container {
    let container = new PIXI.Container();

    //get the maximum predictions by any leafnode for normalization purposes
    let maxPredictions = getMaxPredictions(tree);


    tree.leafNodes.forEach((leafNode, i) => {
        generateSingleCode(container, tree.leafNodes, leafNode, i, maxPredictions)
        generateMisclassificationBlock(container, tree.leafNodes, leafNode, i, maxPredictions);
    })

    generateHorizontalBoundaryLines(container, tree, maxPredictions);
    generateVerticalBoundaryLines(container, tree, maxPredictions);

    return container;
};

function getBarWidth(leafNode: LeafNode, maxPredictions: number) {

    //scale the width based on the total amount of predictions made by this rule. Allows us to visualize the most important rules the most. Not an efficient way to calculate every time
    let predictionCount = leafNode.allRepresentedPredictions!.count;

    //todo: Check why this works weirdly for larger dataset. is it just that there are way too few predictions somehow?

    let percentage = predictionCount / maxPredictions;
    let maxScale = 10;//how much larger is the largest element compared to the smallest element
    let width = horScale * (1 + percentage * (maxScale - 1));
    return width;

    //return horscale; This gives every rule equal width
}


/**
 * Generates a single line of the barcode inside the container
 * @param container The Pixi group we are drawing in. Can use local orientation
 * @param leafNodes: All leafnodes in this tree. Needed when using non-uniform with
 * @param leafNode Current leafNode we will draw as a single vertical line
 * @param leafNumber Current index of the leafNode we are drawing
 * @param totalPredictions Holds the maximum times  a single rule has been used for normalization purposes.
 */
function generateSingleCode(container: PIXI.Container, leafNodes: LeafNode[], leafNode: LeafNode, leafNumber: number, totalPredictions: number) {

    //Put the leafnode and it's representative in a list to draw
    let leafNodesToDraw: LeafNode[] = [];
    leafNodesToDraw.push(leafNode);
    if (leafNode.representedLeafNodes != undefined) {
        leafNodesToDraw = leafNodesToDraw.concat(leafNode.representedLeafNodes);
    }

    //Equal opacity for all
    let targetOpacity = 1 / (leafNodesToDraw.length);
    let minimumOpacity = 0.02;
    let opacity = Math.max(targetOpacity, minimumOpacity); //need to cap it to prevent artifacts, pixijs does not like values lower than 0.02



    let dimRuleRectangleGroup: DimRectangle[] = [];


    let opacitySum = 0;
    for (let leafNodeRep of leafNodesToDraw) {
        if (opacity == minimumOpacity) { //we capped the opacity.  sample the leafnodes to mimic the true opacity.
            opacitySum += targetOpacity;
            if (opacitySum < minimumOpacity) { //Sum is still under the cap, skip it
                continue; //TODO: Resolve this. This is an issue when filtering
            }
            //Over the cap, we draw this leafnode and reduce our sum
            opacitySum -= minimumOpacity;
        }
        let dimRuleRectangle = drawRectangles(container, leafNodes, leafNodeRep, leafNumber, opacity, leafNode, totalPredictions);
        dimRuleRectangleGroup.push(dimRuleRectangle);
    }

    dimRuleRectanglesGroups.push(dimRuleRectangleGroup);

}



function generateHorizontalBoundaryLines(container: PIXI.Container, tree: Tree, maxPredictions: number) {

    boundaryContainers.push(container);

    let lastNode = tree.leafNodes[tree.leafNodes.length - 1];

    let maxX = getLeafX(tree.leafNodes, tree.leafNodes.length - 1, classes.length - 1, maxPredictions) + getBarWidth(lastNode, maxPredictions);


    for (let i = 0; i < tree.leafNodes[0].normalizedRangesPerFeature.length; i++) {

        let x1 = 0;
        let x2 = maxX;
        // let x2 = (tree.leafNodes.length + 1) * (extraHorSpaceBetweenBars + horScale);
        let topY = featureHeight[i] + lineWidth / 2;
        let bottomY = featureHeight[i + 1] - lineWidth / 2 - extraVerSpaceBetweenFeatures
        let boundaryLines = new PIXI.Graphics();
        boundaryLines.moveTo(x1, topY).lineTo(x2, topY);
        boundaryLines.moveTo(x1, bottomY).lineTo(x2, bottomY);
        boundaryLines.stroke({ width: lineWidth, color: lineColor })
        addFeatureDimInteraction(container, i, x1, topY, x2 - x1, bottomY - topY, tree.id);
        container.addChild(boundaryLines);

        if (boundaryLinePerFeatureIndex[i] == undefined) {
            boundaryLinePerFeatureIndex[i] = [];
            boundaryCoordinatesPerFeatureIndex[i] = [];
        }

        boundaryLinePerFeatureIndex[i].push(boundaryLines); //technically does not give you the correct order of containser, but does not matter.
        boundaryCoordinatesPerFeatureIndex[i].push({ x1: x1, x2: x2, y1: topY, y2: bottomY });
    }
}

function generateVerticalBoundaryLines(container: PIXI.Container, tree: Tree, maxPredictions: number) {

    let lineColor = new PIXI.Color(`rgb(20,20,20)`);

    let previousClass = tree.leafNodes[0].class;
    for (let i = 0; i < tree.leafNodes.length; i++) {
        let leafClass = tree.leafNodes[i].class;
        if (leafClass == previousClass) {
            continue;
        }
        //draw the seperation
        previousClass = leafClass;
        let classNumber = classes.indexOf(leafClass);

        let x = getLeafX(tree.leafNodes, i, classNumber, maxPredictions) - extraHorSpaceBetweenClasses / 2;


        let topY = 0;
        let bottomY = featureHeight[featureHeight.length - 1] + featureHeight[1] * 1.2; //*1.2 as the classification has a bit of extra space

        let boundaryLines = new PIXI.Graphics();
        boundaryLines.moveTo(x, bottomY).lineTo(x, topY);
        boundaryLines.stroke({ width: extraHorSpaceBetweenClasses / 2, color: lineColor })

        container.addChild(boundaryLines);
    }
}


/**
 * Holds the rectangles that allow us to dim per featureindex. Need 2 of them to highlight ranges
 */
let dimRectanglePerFeaturesTop: [PIXI.Graphics[]] = [[]];
let dimRectanglePerFeaturesBottom: [PIXI.Graphics[]] = [[]];

/**
 * Adds the rectangles for the dimming interaction.
 * @param container 
 * @param featureNumber 
 * @param leftX 
 * @param topY 
 * @param width 
 * @param height 
 */
function addFeatureDimInteraction(container: PIXI.Container, featureNumber: number, leftX: number, topY: number, width: number, height: number, treeIndex: number) {
    let rectInteractiveTop = new PIXI.Graphics();
    rectInteractiveTop.rect(leftX, topY, width, height);
    rectInteractiveTop.fill({ color: "0xffffff", alpha: 1 })

    rectInteractiveTop.alpha = 0;

    rectInteractiveTop.interactive = true;
    rectInteractiveTop.hitArea = new PIXI.Rectangle(leftX, topY, width, height);

    let rectInteractiveBottom = new PIXI.Graphics();
    rectInteractiveBottom.rect(leftX, topY, width, height);
    rectInteractiveBottom.fill({ color: "0xffffff", alpha: 1 })

    rectInteractiveBottom.alpha = 0;

    rectInteractiveBottom.interactive = true;
    rectInteractiveBottom.hitArea = new PIXI.Rectangle(leftX, topY, width, height);

    if (dimRectanglePerFeaturesTop[featureNumber] == undefined) {//make sure it's initialized
        dimRectanglePerFeaturesTop[featureNumber] = [];
        dimRectanglePerFeaturesBottom[featureNumber] = [];
    }
    dimRectanglePerFeaturesTop[featureNumber].push(rectInteractiveTop);
    dimRectanglePerFeaturesBottom[featureNumber].push(rectInteractiveBottom);

    rectInteractiveTop.onmouseover = function (mouseData) {
        focusFeature(featureNumber);
        selectClusterProjection(treeIndex);
    }
    rectInteractiveTop.onmouseleave = function (mouseData) {
        unfocusFeature(featureNumber);
        selectClusterProjection();
    }

    rectInteractiveBottom.onmouseover = function (mouseData) {
        focusFeature(featureNumber);
        selectClusterProjection(treeIndex);
    }
    rectInteractiveBottom.onmouseleave = function (mouseData) {
        unfocusFeature(featureNumber);
        selectClusterProjection();
    }
    container.addChild(rectInteractiveTop);
    container.addChild(rectInteractiveBottom);
}

/**
 * Highlights the selected feature
 * @param featureIndex 
 */
export function highlightBarcodeFeature(featureIndex: number) {
    if (featureIndex > boundaryLinePerFeatureIndex.length) {
        throw new Error(`Cannot dim barcode feature ${featureIndex}. Only ${boundaryLinePerFeatureIndex} features are in the barcode`)
    }
    updateBoundaryLines(featureIndex)
}

/**
 * 
 * @param featureToDim Holds which features to highlight
 */
function updateBoundaryLines(featureToHighlight: number) {
    //go through each container, and update the line
    for (let containerIndex = 0; containerIndex < boundaryContainers.length; containerIndex++) {
        let boundaryContainer = boundaryContainers[containerIndex];
        for (let featureIndex = 0; featureIndex < boundaryLinePerFeatureIndex.length; featureIndex++) {
            //Clear last line, can't simply adapt as it then doesn't update
            let lineGraphics = boundaryLinePerFeatureIndex[featureIndex][containerIndex];
            lineGraphics.clear();
            lineGraphics.parent.removeChild(lineGraphics); //remove from the parent so we don't keep adding them

            //make new line
            let x1 = boundaryCoordinatesPerFeatureIndex[featureIndex][containerIndex].x1;
            let x2 = boundaryCoordinatesPerFeatureIndex[featureIndex][containerIndex].x2;
            let y1 = boundaryCoordinatesPerFeatureIndex[featureIndex][containerIndex].y1;
            let y2 = boundaryCoordinatesPerFeatureIndex[featureIndex][containerIndex].y2;
            let strokeWidth = lineWidth;
            if (featureToHighlight == featureIndex) { //increase the width, and use the empty whitespace between rows to not overlap the data
                strokeWidth = lineWidth * 4;
                y1 -= lineWidth * 1.5; //not sure why 1.5 but works.
                y2 += lineWidth * 1.5;
                lineColor = new PIXI.Color("rgb(0,0,0)");
            }

            lineGraphics = new PIXI.Graphics();
            lineGraphics.moveTo(x1, y1).lineTo(x2, y1);
            lineGraphics.moveTo(x1, y2).lineTo(x2, y2);
            lineGraphics.stroke({ width: strokeWidth, color: lineColor })


            boundaryLinePerFeatureIndex[featureIndex][containerIndex] = lineGraphics;
            boundaryContainer.addChild(lineGraphics); //need to readd it so pixi knows it needs to update

        }
    }
}


/**
 * Dims all barcode rules that do not lie in the specified interval
 * @param featureRanges 
 */
export function dimBarcodeRules(featureRanges: [number, number][], classificationsToShow: boolean[][]): void {

    for (let dimRuleGroup of dimRuleRectanglesGroups) {

        let maxRules = dimRuleGroup.length;
        //0.02 baseopacity as pixi gets artifacts otherwise
        let targetOpacity = Math.max(0.02, 1 / maxRules);

        for (let dimRule of dimRuleGroup) {
            let opacity = 0; //hide inactive rules
            if (isRuleActive(dimRule, featureRanges, classificationsToShow)) {
                opacity = targetOpacity;
            }
            dimRule.rectangles.forEach((rectangle) => rectangle.alpha = opacity);
        }
    }
}

function areAllClassificationsSelected(classificationsSelected: boolean[][]): boolean {

    for (let originalClassI = 0; originalClassI < classes.length; originalClassI++) {
        for (let targetClassI = 0; targetClassI < classes.length; targetClassI++) {
            if (classificationsSelected[originalClassI][targetClassI] == false) {
                return false;
            }
        }
    }
    return true;
}



function drawRectangles(container: PIXI.Container, leafNodes: LeafNode[], leafNode: LeafNode, leafNumber: number, opacity: number, representative: LeafNode, maxPredictions: number): DimRectangle {

    let data = leafNode.normalizedRangesPerFeature;
    let dimRectangles = [];
    let rectanglesObject = new PIXI.Graphics();

    let classNumber = classes.indexOf(leafNode.class);

    let x = getLeafX(leafNodes, leafNumber, classNumber, maxPredictions);
    let width = getBarWidth(representative, maxPredictions); //we use the width of the representative

    for (let featureI = 0; featureI < data.length; featureI++) {
        let d = data[featureI];
        let y = featureHeight[featureI] + d[0] * getFeatureCoordinateRange(featureI);
        let height = getBarHeight(d, featureI);
        let color = new PIXI.Color(featureColors(featureI / data.length));

        //use a minimumheight.
        let maxHeight = getBarHeight([0, 1], featureI);
        height = Math.max(height, maxHeight * 0.05);
        //if this goes over the height of the feature, move the y coordinate
        if ((y + height) > (featureHeight[featureI] + maxHeight)) {
            y = featureHeight[featureI] + maxHeight - height;
        }

        rectanglesObject.rect(x, y, width, height);
        rectanglesObject.fill({ color: color })

        dimRectangles.push(rectanglesObject);
    }
    rectanglesObject.alpha = opacity; //Set alpha instead of opacity so we can change it when needed when hovering.
    container.addChild(rectanglesObject);

    let allPredictions = leafNode.allPredictions!;

    let dimRule = { normalizedRanges: data, predictions: allPredictions, rectangles: dimRectangles };
    return dimRule;
}

function getClassColor(className: string): string {
    const index = classes.indexOf(className);
    return classColors(index / classes.length);
}


function setFeatureHeight(trees: Tree[]): void {
    featureHeight = [];
    let featureCount = getFeatureCount(trees);

    //FeatureValues sum up to 1. Using it to divide over available space
    let totalHeight = featureCount * (verScale + 1 - extraVerSpaceBetweenFeatures)
    let featureSumPos = 0;
    let featureSumVal = 0;

    for (let i = 0; i < featureCount; i++) {
        featureHeight[i] = featureSumPos; //starts at the current sum, calculate new start for next iteration

        console.log("Not using actualy feature power. Still a placeholder");
        // let featureValue = 1 / (Math.pow(2, i + 1));//Replace by actual value
        let featureValue = 1 / featureCount;//Equal height

        if (i == (featureCount - 1)) { //temp, making sure we fill it up.
            featureValue = 1 - featureSumVal;
        }
        featureSumVal += featureValue
        //
        let position = featureValue * totalHeight + extraVerSpaceBetweenFeatures;
        featureSumPos += position;
    }

    featureHeight[featureCount] = featureSumPos; //set the last feature so we know where the space ends
}



function getBarHeight(data: [number, number], featureNumber: number): number {
    // if (isFullBar(data)) {
    //     return 0;
    // }
    let height = Math.max(0.02, (data[1] - data[0]) * getFeatureCoordinateRange(featureNumber))
    return height;
}


//returns the amount of pixels available for the feature with number {featureNumber]}
function getFeatureCoordinateRange(featureNumber: number) {
    return featureHeight[featureNumber + 1] - featureHeight[featureNumber] - extraVerSpaceBetweenFeatures;
}

function isFullBar(data: [number, number]): boolean {
    if (data[0] == 0 && data[1] == 1) {
        return true;
    }
    return false;
}

/**
 * Returns whether testRanges overlaps with targetRanges for each feature.
 * 
 * @param testRanges 
 * @param targetRanges 
 */
function withinRange(testRanges: [number, number][], targetRanges: [number, number][]): boolean {
    if (testRanges.length != targetRanges.length) {
        console.error(`testranges ${testRanges.length} does not have the same length as targetranges ${targetRanges.length}`)
    }
    for (let i = 0; i < testRanges.length; i++) {
        let testRange = testRanges[i];
        let targetRange = targetRanges[i];
        //if at least 1 feature does not overlap, it is not within range
        if (testRange[0] > targetRange[1]) { //testRange starts after target
            return false;
        }
        if (testRange[1] < targetRange[0]) { //testRange ends before target
            return false;
        }

    }
    return true;
}

let dimClassRectangles: {
    originClassI: number;
    targetClassI: number;
    rectangle: PIXI.Graphics
}[] = []

/**
 * 
 * @param container 
 * @param leafNodes all nodes in this cluster
 * @param leafNode One (representative) leafnode for which we want to generate the column
 * @param leafNumber index of the column
 * @param maxPredictions maximum amount of predictions present in this tree within 1 leafnode.
 */
function generateMisclassificationBlock(container: PIXI.Container<PIXI.Container>, leafNodes: LeafNode[], leafNode: LeafNode, leafNumber: number, maxPredictions: number) {

    let predictions = leafNode.allRepresentedPredictions!;

    //get the classnumber of this leafnode. All representative must have the same classnumber by definition
    let classNumber = classes.indexOf(leafNode.class);

    //get predictions per misclassification
    let predictionPerOriginClass: number[] = [];

    for (let originClassI = 0; originClassI < classes.length; originClassI++) {
        predictionPerOriginClass[originClassI] = predictions[originClassI][classNumber]; //always classified towards the class of the leafNode
    }

    let x = getLeafX(leafNodes, leafNumber, classNumber, maxPredictions);
    let width = getBarWidth(leafNode, maxPredictions);
    let totalHeight = getFeatureCoordinateRange(0); //same height as feature 0
    let y = featureHeight[featureHeight.length - 1] + totalHeight * 1.2; //start at the bottom and leave a bit of space

    for (let originClassI = 0; originClassI < classes.length; originClassI++) {
        let predictionCount = predictionPerOriginClass[originClassI];
        if (predictionCount == 0) { //Don't draw it if there were no classifications from the class.
            continue;
        }
        let targetColor = getClassColor(classes[originClassI]);
        let percentageOfMax = (predictionCount / maxPredictions);


        let height = percentageOfMax * totalHeight;
        height = Math.max(height, 3);//at least 3 units high such that you can always at least see something
        y -= height

        let rectangleObject = new PIXI.Graphics();
        rectangleObject.rect(x, y, width, height);
        rectangleObject.fill({ color: targetColor, alpha: 1 })

        //get values for the tooltip on hover
        let nicePercentage: string = Math.ceil(predictionCount / predictions.count * 1000) / 10 + "%"; //single decimal
        let tooltipText = "";
        if (originClassI == classNumber) {
            tooltipText = "" + classes[originClassI] + " is correctly classified in " + nicePercentage + " \nof all cases in these decision paths";
        } else {
            tooltipText = "" + classes[originClassI] + " is misclassified as " + classes[classNumber] + " in " + nicePercentage + " \nof all cases in these decision paths";
        }

        rectangleObject.interactive = true;
        rectangleObject.onmouseover = (eventData) => {
            classHovered(originClassI, classNumber);
        }
        rectangleObject.onmousemove = (eventData) => {
            showBarCodeToolTip(eventData.global.x, eventData.global.y, tooltipText);
        }
        rectangleObject.onmouseleave = () => {
            hideBarCodeToolTip();
            classStopHover(originClassI, classNumber);
        }

        dimClassRectangles.push({ originClassI: originClassI, targetClassI: classNumber, rectangle: rectangleObject });
        container.addChild(rectangleObject);
    }
}

function getLeafX(leafNodes: LeafNode[], leafNumber: number, classNumber: number, maxPredictions: number): number {
    //not efficient, but works
    let x = 0;
    for (let i = 0; i < leafNumber; i++) {
        let leafNode = leafNodes[i];
        let width = getBarWidth(leafNode, maxPredictions);
        x += (extraHorSpaceBetweenBars + width);
    }

    //add the extra space for class seperation
    x += extraHorSpaceBetweenClasses * classNumber

    // let x = leafNumber * (extraHorSpaceBetweenBars + width) + extraHorSpaceBetweenClasses * classNumber;
    return x;
}

/**
 * Returns a value between 0 and 1
 * @param predictions 
 * @param classificationsSelected 
 * @param countZeroPrediction If true, treats a prediction count of 0 as 100%.
 * @returns 
 */
function getClassificationPercentage(predictions: Predictions, classificationsSelected: boolean[][], countZeroPrediction: boolean): number {
    let total = predictions.count;
    if (total == 0) {
        return countZeroPrediction ? 1 : 0;
    }
    let sum = 0;

    for (let originalClassI = 0; originalClassI < classes.length; originalClassI++) {
        for (let targetClassI = 0; targetClassI < classes.length; targetClassI++) {
            if (classificationsSelected[originalClassI][targetClassI] == false) {
                continue;
            }
            let count = predictions[originalClassI][targetClassI];
            sum += count;
        }
    }

    return sum / total;
}

/**
 * Sets the "allPrediction" matrix in the trees, does not recurse into all representatives yet
 * @param trees 
 */
function setAllPredictions(trees: Tree[]) {
    trees.forEach(tree => {
        tree.leafNodes.forEach(leafNode => {
            //initialize the predictions, make sure it's complete.
            let predictions: Predictions = { count: 0 };
            for (let originalClassI = 0; originalClassI < classes.length; originalClassI++) {
                predictions[originalClassI] = { count: 0 }
                for (let targetClassI = 0; targetClassI < classes.length; targetClassI++) {
                    predictions[originalClassI][targetClassI] = 0;
                }
            }

            //add the correct predictions
            let classNumber = classes.indexOf(leafNode.class)
            predictions.count += leafNode.correctPredictions
            predictions[classNumber].count += leafNode.correctPredictions;
            predictions[classNumber][classNumber] += leafNode.correctPredictions

            //add the incorrect predictions
            for (let sampleClassI = 0; sampleClassI < classes.length; sampleClassI++) {
                if (leafNode.wrongPredictions == null || leafNode.wrongPredictions[sampleClassI] == undefined) {
                    continue; //no data
                }
                predictions.count += leafNode.wrongPredictions[sampleClassI].count;

                predictions[sampleClassI].count += leafNode.wrongPredictions[sampleClassI].count;
                for (let misclassifiedIntoClassI = 0; misclassifiedIntoClassI < classes.length; misclassifiedIntoClassI++) {
                    if (leafNode.wrongPredictions[sampleClassI][misclassifiedIntoClassI] == undefined) {
                        continue;//no data
                    }
                    predictions[sampleClassI][misclassifiedIntoClassI] += leafNode.wrongPredictions[sampleClassI][misclassifiedIntoClassI];
                }
            }
            leafNode.allPredictions = predictions;
        })
    })
}

/**
 * Should only be called after allpredictions is set. Could be merged into above
 * @param trees 
 */
function setRepresentativePredictions(trees: Tree[]) {
    trees.forEach(tree => {
        tree.leafNodes.forEach(leafNode => {

            //initialize the predictions, make sure it's complete.
            let predictions: Predictions = { count: 0 };
            for (let originalClassI = 0; originalClassI < classes.length; originalClassI++) {
                predictions[originalClassI] = { count: 0 }
                for (let targetClassI = 0; targetClassI < classes.length; targetClassI++) {
                    predictions[originalClassI][targetClassI] = 0;
                }
            }
            //get the representatives
            let leafNodes: LeafNode[] = [];
            if (leafNode.representedLeafNodes) {
                leafNodes = [...leafNode.representedLeafNodes];
            }
            leafNodes.push(leafNode); //add itself

            for (let leafNode of leafNodes) {
                predictions.count += leafNode.allPredictions!.count;
                for (let originalClassI = 0; originalClassI < classes.length; originalClassI++) {
                    predictions[originalClassI].count += leafNode.allPredictions![originalClassI].count;
                    for (let targetClassI = 0; targetClassI < classes.length; targetClassI++) {
                        predictions[originalClassI][targetClassI] += leafNode.allPredictions![originalClassI][targetClassI];
                    }
                }
            }
            leafNode.allRepresentedPredictions = predictions;
        })
    })
}


function isRuleActive(
    dimRule: { normalizedRanges: [number, number][]; predictions: Predictions; rectangles: PIXI.Graphics[]; },
    featureRanges: [number, number][],
    classificationsToShow: boolean[][]): boolean {

    let allSelected = areAllClassificationsSelected(classificationsToShow);

    let ranges = dimRule.normalizedRanges;
    if (!withinRange(ranges, featureRanges)) {//dim if not in range.
        return false;
    }
    if (allSelected) { //no need to filter further
        return true;
    }
    let classificationPercentage = getClassificationPercentage(dimRule.predictions, classificationsToShow, false);
    if (classificationPercentage == 0) {
        return false;
    }
    return true;
}