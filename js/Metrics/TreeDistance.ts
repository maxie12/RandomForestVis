import { LeafNode } from "../types";


export function getDistanceMatrixFromLeafs(leafs: LeafNode[]): number[][] {

    let distanceMatrix: number[][] = [];

    for (let i = 0; i < leafs.length; i++) {
        let leafNode1 = leafs[i];
        for (let j = 0; j < leafs.length; j++) {
            let leafNode2 = leafs[j];
            let distance = getDistanceBetweenLeafNodes(leafNode1, leafNode2);


            if (distanceMatrix[i] == undefined) {
                distanceMatrix[i] = [];//TODO: Do nicely
            }
            distanceMatrix[i][j] = distance;
        }
    }
    return distanceMatrix;

}

export function getDistanceBetweenLeafNodes(leafNode1: LeafNode, leafNode2: LeafNode): number {
    if (leafNode1.normalizedRangesPerFeature.length == 0) {
        console.error("LeafNode1 does not have ranges.")
        console.error(leafNode1);
        return NaN;
    }

    let totalDistance = 0;
    for (let featureI = 0; featureI < leafNode1.normalizedRangesPerFeature.length; featureI++) {
        let distance = getDistanceBetweenIntervals(leafNode1.normalizedRangesPerFeature[featureI], leafNode2.normalizedRangesPerFeature[featureI]);;
        totalDistance += distance
    }
    return totalDistance / leafNode1.normalizedRangesPerFeature.length;
}


/**
 * Gets a normalized distance between the two intervals based on overlap. Needs to be updated/precomputed as part of clustering.
 * @param interval1 
 * @param interval2 
 */
export function getDistanceBetweenIntervals(interval1: [number, number], interval2: [number, number]): number {
    //startInterval is the one with the earliest startpoint.

    let startInterval = interval1[0] < interval2[1] ? interval1 : interval2
    let otherInterval = interval1[0] < interval2[1] ? interval2 : interval1

    if (startInterval[1] < otherInterval[0]) {
        //start ends before other begins
        return 1; //maximum distance
    }

    //overlap is from starting point to first endpoint.
    let firstEndPoint = Math.min(startInterval[1], otherInterval[1]);
    let overlap = firstEndPoint - otherInterval[0];
    let totalRange = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]);

    if (totalRange == 0) {
        return 0; //complete overlap. Prevent division by 0 errors
    }

    //near-perfect overlap gives a score of 1. Need to divide totalRange by 2 as we have 2 ranges.
    let overlapScore = (overlap / (totalRange / 2));
    //Invert to give a distance score
    if (Number.isNaN(overlapScore)) {
        console.error("distance between " + interval1 + " and " + interval2 + " results in a Nan score");
    }
    return 1 - overlapScore;
}

