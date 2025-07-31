import { dehoverClusterMainView, hoverClusterMainView, selectClusterMainView } from "./index";
import { dimBarcodeRules, highlightBarcodeFeature } from "./Barcode/Barcode";
import { dimFeatureInFeatureDistributionPlot, scrollToFeature, undimFeatureInFeatureDistributionPlot } from "./Sidebar/FeatureChart";
import { dimFeaturePlot, dimFeaturePlotFeature, undimFeaturePlotFeature } from "./Featureplot/Featureplot";
import { dimNodeLinkPaths, highlightNodeLinkFeature, selectClusterNodeLink, unHighlightNodeLinkFeature } from "./NodeLink";
import { selectClusterProjection } from "./Projection";
import { dimClasses } from "./Sidebar/ClassChart";
import { Predictions } from "./types";

let featureCount = 0;
let currentFeatureDistributions: [number, number][] = []; //holds the ranges of features currently selected
let currentOriginalFeatureDistributions: [number, number][] = []; //holds the ranges of non-normalized features currently selected

let currentClassificationsSelected: boolean[][] = []; //holds the (mis)classifications currently selected
let hoverOriginClass: number; //the id of the class we are hovering
let hoverTargetClass: number; //The id of the misclassification class if present

export function initClassInteraction(classCount: number) {
    currentClassificationsSelected = []
    for (let i = 0; i < classCount; i++) {
        currentClassificationsSelected[i] = []
        for (let j = 0; j < classCount; j++) {
            currentClassificationsSelected[i][j] = true;
        }
    }
}

export function initFeatureDistribution(amountOfFeatures: number) {
    currentFeatureDistributions = [];
    for (let i = 0; i < amountOfFeatures; i++) {
        currentFeatureDistributions[i] = [0, 1]; //start with everything visible
    }
    featureCount = amountOfFeatures
}

export function initInteractionOriginalRanges(originalRanges: number[][]) {
    for (let i = 0; i < originalRanges.length; i++) {
        currentOriginalFeatureDistributions[i] = [originalRanges[i][0], originalRanges[i][1]]
    }
}


export function classHovered(originClass: number, targetClass: number = NaN) {
    hoverOriginClass = originClass;
    hoverTargetClass = targetClass;
    updateAll();
}

export function classStopHover(originClass: number, targetClass: number = NaN) {
    if (hoverOriginClass == originClass && (hoverTargetClass == targetClass || (isNaN(hoverTargetClass) && isNaN(targetClass)))) {
        //if it's not the same, this is a later one. Ignore.
        hoverOriginClass = NaN;
        hoverTargetClass = NaN;
    }
    updateAll();
}

export function toggleOriginalClassSelected(class1: number) {
    let classFullySelected = true;
    let classCount = currentClassificationsSelected.length

    if (areAllActiveClassesSelected()) {
        deselectAllClasses();
    }
    for (let j = 0; j < classCount; j++) {
        if (!currentClassificationsSelected[class1][j]) {
            classFullySelected = false;
        }
    }


    for (let j = 0; j < classCount; j++) {
        if (classFullySelected) { //if it was selected, deselect it
            currentClassificationsSelected[class1][j] = false;
        } else {//otherwise select is
            currentClassificationsSelected[class1][j] = true;
        }

    }
    updateClasses();
}

let classificationsPresent: boolean[][];
let classCount: number;
export function initInteractionClasses(_classCount: number, allPredictions: Predictions) {
    classCount = _classCount;
    classificationsPresent = [];
    for (let i = 0; i < classCount; i++) {
        classificationsPresent[i] = [];
        for (let j = 0; j < classCount; j++) {
            classificationsPresent[i][j] = (allPredictions[i][j] > 0)
        }
    }
}

/**
 * toggles the selected class from being selected or not. Starts with everything selected
 */
export function toggleClassificationSelected(class1: number, class2: number) {
    if (areAllActiveClassesSelected()) {
        deselectAllClasses();
        currentClassificationsSelected[class1][class2] = true;
    } else {
        currentClassificationsSelected[class1][class2] = !currentClassificationsSelected[class1][class2];
    }


    updateClasses()
}

function areAllActiveClassesSelected(): boolean {
    for (let i = 0; i < classCount; i++) {
        for (let j = 0; j < classCount; j++) {
            if (!classificationsPresent[i][j]) {
                continue;
            }
            //class is selected
            if (currentClassificationsSelected[i][j] == false) {
                return false;
            }
        }
    }
    //all classifications that are present are selected
    return true;
}

function areAllActiveClassesDeselected(): boolean {
    for (let i = 0; i < classCount; i++) {
        for (let j = 0; j < classCount; j++) {
            if (!classificationsPresent[i][j]) {
                continue;
            }
            //class is selected
            if (currentClassificationsSelected[i][j] == true) {
                return false;
            }
        }
    }
    //all classifications that are present are deselected
    return true;
}

function updateClasses() {
    if (areAllActiveClassesSelected() || areAllActiveClassesDeselected()) {//ensure that classes that can't be clicked on don't interfer
        selectAllClasses();
    }

    updateAll();
}


function selectAllClasses() {
    for (let i = 0; i < classCount; i++) {
        for (let j = 0; j < classCount; j++) {
            currentClassificationsSelected[i][j] = true
        }
    }
}
function deselectAllClasses() {
    for (let i = 0; i < classCount; i++) {
        for (let j = 0; j < classCount; j++) {
            currentClassificationsSelected[i][j] = false
        }
    }
}

/**
 *  
 * @param featureRanges Array of ranges that are selected.
 * @param originalFeatureRanges Array of non-normalized ranges that are selected.
 */
export function updateFeatureDistributionSelection(featureRanges: [number, number][], orginalFeatureRanges: [number, number][]) {
    currentFeatureDistributions = featureRanges;
    currentOriginalFeatureDistributions = orginalFeatureRanges;
    updateAll()
}

function getClassesToVisualize() {
    let classesToVisualize: boolean[][] = structuredClone(currentClassificationsSelected)
    if (Number.isNaN(hoverOriginClass) || hoverOriginClass == undefined) { //no class is hovered, can use as is.
        return classesToVisualize;
    }

    //class is hovered
    if (areAllActiveClassesSelected()) { //If all classes are selected, we need to deselect everything and only highlight those hovered
        for (let i = 0; i < classesToVisualize.length; i++) {
            for (let j = 0; j < classesToVisualize.length; j++) {
                classesToVisualize[i][j] = false
            }
        }
    }
    //if we are hovering over the misclassification, only need to show 1 classcombination.
    if (!isNaN(hoverTargetClass)) {
        classesToVisualize[hoverOriginClass][hoverTargetClass] = true
        return classesToVisualize;
    }
    //otherwise show the entire row
    for (let j = 0; j < classesToVisualize[hoverOriginClass].length; j++) {
        classesToVisualize[hoverOriginClass][j] = true
    }
    return classesToVisualize;
}

export function updateAll() {
    let classesToVisualize = getClassesToVisualize()

    dimBarcodeRules(currentFeatureDistributions, classesToVisualize);
    dimFeaturePlot(currentFeatureDistributions, classesToVisualize);
    //splitpoints in the tree are not normalized, so we use the original feature range to dim paths

    dimNodeLinkPaths(currentOriginalFeatureDistributions, classesToVisualize);

    dimClasses(classesToVisualize)
}

export function focusFeature(featureIndex: number) {
    highlightNodeLinkFeature(featureIndex);
    highlightBarcodeFeature(featureIndex);
    scrollToFeature(featureIndex);
    dimOtherFeatures(featureIndex);
}

export function unfocusFeature(featureIndex: number) {
    unHighlightNodeLinkFeature(featureIndex);
    highlightBarcodeFeature(-1);
    undimFeatures();
}


/**
 * Dim all features except for the one in featureIndex
 * @param featureIndex 
 */
function dimOtherFeatures(featureIndex: number) {
    for (let i = 0; i < featureCount; i++) {
        if (i == featureIndex) {//don't dim this features
            continue;
        }
        dimFeatureNoRerender(i);
    }
}

function undimFeatures() {
    for (let i = 0; i < featureCount; i++) {
        undimFeatureNoRerender(i);
        undimFeatureInFeatureDistributionPlot(i);
    }
}

let prevId: number | undefined = undefined
export function selectCluster(treeId: number) {
    //TODO: For some reason this is getting called twice from both barplot and featureplot. Not sure why. 
    //This prevent deselecting from from fully working. Workout to select an intial one?
    if (prevId == treeId) {
        return;
    }
    prevId = treeId;

    selectClusterNodeLink(treeId);
    selectClusterMainView(treeId);
    selectClusterProjection(treeId);
}

export function highlightCluster(treeId: number) {
    hoverClusterMainView(treeId);
}


export function unhighlightCluster(treeId: number) {
    dehoverClusterMainView(treeId);
}

function dimFeatureNoRerender(featureIndex: number) {
    dimFeatureInFeatureDistributionPlot(featureIndex);
    dimFeaturePlotFeature(featureIndex)
}

function undimFeatureNoRerender(featureIndex: number) {
    undimFeatureInFeatureDistributionPlot(featureIndex);
    undimFeaturePlotFeature(featureIndex)
}
