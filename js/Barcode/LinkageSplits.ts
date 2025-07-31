import { link, map } from "d3";
import { LinkageSplitPoint, LinkageTree } from "../types";


function convertToTree(linkageNodes: LinkageSplitPoint[]): LinkageTree {

    let sortedNodes = linkageNodes.sort((a, b) => b.distance - a.distance);

    let idMap = new Map<number, LinkageTree>()

    for (let node of sortedNodes) {
        let treeNode: LinkageTree = {
            id: node.id,
            distance: node.distance,
            child1: node.children[0].id,
            child2: node.children[1].id
        }
        idMap.set(treeNode.id, treeNode);
    }

    let root = idMap.get(sortedNodes[0].id);
    if (root == undefined) {
        throw new Error("Root cannot be undefined")
    }

    recursiveTreeBuild(root, idMap);
    return root;
}

function selectNodes(root: LinkageTree, targetAmount: number): LinkageTree[] {
    let selectedNodes: LinkageTree[] = []; //sorted list by distance of nodes
    selectedNodes.push(root); //initialize using the root of the tree

    while (selectedNodes.length != targetAmount) {
        //remove the first node and add it's children
        let tree = selectedNodes[0];
        selectedNodes.shift();
        if (typeof tree.child1 != "number") {
            selectedNodes.push(tree.child1);
        }
        if (typeof tree.child2 != "number") {
            selectedNodes.push(tree.child2);
        }
        //resort the list by largest distance first. Inefficient but works
        selectedNodes.sort((a, b) => b.distance - a.distance);
    }
    return selectedNodes;
}


function recursiveTreeBuild(currentRoot: LinkageTree, mapping: Map<number, LinkageTree>) {
    if (currentRoot == undefined) {
        throw new Error("TreeNode is not defined");
    }
    if (typeof currentRoot.child1 != "number") {
        throw new Error("child1 was already assigned");
    }
    if (typeof currentRoot.child2 != "number") {
        throw new Error("child2 was already assigned");
    }

    if (mapping.has(currentRoot.child1)) {
        currentRoot.child1 = mapping.get(currentRoot.child1)!; //! to denote type is not null
        recursiveTreeBuild(currentRoot.child1, mapping)
    }
    if (mapping.has(currentRoot.child2)) {
        currentRoot.child2 = mapping.get(currentRoot.child2)!; //! to denote type is not null
        recursiveTreeBuild(currentRoot.child2, mapping)
    }

}

function getLeafIds(node: LinkageTree): number[] {
    let leafIds: number[] = [];
    if (typeof node.child1 == "number") {
        leafIds.push(node.child1);
    } else {
        let childIds = getLeafIds(node.child1)
        leafIds = leafIds.concat(childIds);
    }

    if (typeof node.child2 == "number") {
        leafIds.push(node.child2);
    } else {
        let childIds = getLeafIds(node.child2)
        leafIds = leafIds.concat(childIds);
    }
    return leafIds;
}
function findCenterNode(representativeIds: number[]): number {
    representativeIds.sort();
    //take the first one as the representatives
    let representId = representativeIds[Math.floor(representativeIds.length / 2)];
    return representId;
}

