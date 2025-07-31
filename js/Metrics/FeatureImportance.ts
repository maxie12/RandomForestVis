import { getFeatureCount } from "../Barcode/Tree";
import { getDistanceBetweenIntervals } from "./TreeDistance";
import { LeafNode, Tree } from "../types";

/**
 * Returns the feature importance for each feature in an array. More important features have high scores
 * @param trees 
 */
export function getFeatureImportance(trees: Tree[], method: "featureRange" | "contributions" | "mergeAgreement"): number[] {
    console.log("Implement these properly for height of featureplot")

    if (method === "featureRange") {
        return getFrequencyImportance(trees);
    }
    if (method === "contributions") {
        return getFeatureContributions(trees)
    }
    if (method === "mergeAgreement") {
        return getMergeAgreement(trees)
    }

    console.error("Method ${method} is not yet implemented.");
    return []
}


function getFrequencyImportance(trees: Tree[]): number[] {
    let featureCount = trees[0].leafNodes[0].normalizedRangesPerFeature.length;


    //Holds the total range per feature for each class
    //Smallest ranges hold the most weight
    let sumRangePerFeature: number[] = [];

    for (let i = 0; i < featureCount; i++) {
        sumRangePerFeature[i] = 0;
    }

    for (let tree of trees) {
        for (let leafNode of tree.leafNodes) {
            for (let i = 0; i < featureCount; i++) {
                let range = leafNode.normalizedRangesPerFeature[i];
                sumRangePerFeature[i] += range[1] - range[0];
            }
        }
    }
    //sumRangePerFeature now contains total ranges. 
    //invert scores to ensure that more important features (smaller ranges) have higher scores

    sumRangePerFeature.forEach(d => -d);

    return sumRangePerFeature;
}





function getFeatureContributions(trees: Tree[]): number[] {
    let featureContributions: number[] = [];
    let featureCount = getFeatureCount(trees);



    return featureContributions;
}

function getMergeAgreement(trees: Tree[]): number[] {
    let featureCount = trees[0].leafNodes[0].normalizedRangesPerFeature.length;


    //holds average agreement per feature
    let averageAgreementPerFeature: number[] = [];


    for (let featureI = 0; featureI < featureCount; featureI++) {
        let sumAgreement = 0;
        for (let tree of trees) {
            let averageTreeAgreement = getAverageTreeAgreement(tree, featureI);
            sumAgreement += averageTreeAgreement;
        }
        averageAgreementPerFeature[featureI] = 1-(sumAgreement / trees.length);
    }
    return averageAgreementPerFeature;
}
function getAverageTreeAgreement(tree: Tree, featureI: number): number {
    let sumTreeAgreement = 0;
    let emptyNodes = 0;
    for (let leafNode of tree.leafNodes) {
        if (leafNode.representedLeafNodes == undefined) {
            emptyNodes++
            continue;
        }
        let averageAgreement = getAverageLeafAgreement(leafNode, featureI);
        sumTreeAgreement += averageAgreement;
    }

    let averageAgreement = sumTreeAgreement / (tree.leafNodes.length-emptyNodes);//normalize for amount of leaves
    return averageAgreement;
}


function getAverageLeafAgreement(leafNode: LeafNode, featureI: number): number {
    let sumDistance = 0;

    let interval1 = leafNode.normalizedRangesPerFeature[featureI];
    for (let representedNode of leafNode.representedLeafNodes) {
        let interval2 = representedNode.normalizedRangesPerFeature[featureI];
        let distance = getDistanceBetweenIntervals(interval1, interval2);
        sumDistance += distance;
    }
    let averageDistance = sumDistance / leafNode.representedLeafNodes.length;//normalize for amount of representatives
    return 1 - averageDistance;
}