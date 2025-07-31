//contains method that work on the tree
import { getDistanceBetweenLeafNodes } from "../Metrics/TreeDistance";
import { LeafNode, Tree } from "../types";



export function getFeatureCount(trees: Tree[]): number {
    return trees[0].leafNodes[0].normalizedRangesPerFeature.length;
}

export function mapLeafNodes(treeRep: Tree, treeToMap: Tree) {
    for (let leafNode of treeToMap.leafNodes) {
        if (leafNode == undefined) {
            console.error(treeToMap);
        }
        let bestMatch: LeafNode | null = null;
        let bestMatchDistance = 9999999999;

        for (let repLeafNode of treeRep.leafNodes) {
            if (repLeafNode.class != leafNode.class) {
                continue; //only map equal classes.
            }

            let distance = getDistanceBetweenLeafNodes(leafNode, repLeafNode);
            if (distance < bestMatchDistance) {
                bestMatch = repLeafNode;
                bestMatchDistance = distance;
            }
        }
        if (bestMatch == null) {
            console.error("No match found for leafNode: " + leafNode);
            return;
        }

        if (bestMatch.representedLeafNodes == undefined) {
            bestMatch.representedLeafNodes = [];
        }
        bestMatch.representedLeafNodes.push(leafNode);
    }
}