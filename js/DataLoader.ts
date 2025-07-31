import { LeafNode, Tree, DTree, DTreeNode, FeaturePlotData, Cluster, Predictions } from "./types";

export function readDtreeCSVData(csvData: d3.DSVRowArray<string>, classCount: number): Map<number, DTree> {
	let dTreeMap = new Map<number, DTree>();
	for (let row of csvData) {
		let treeId = parseInt(row.tree_id);
		let dtree: DTree;

		if (dTreeMap.has(treeId)) {
			dtree = dTreeMap.get(treeId) as DTree;
		} else {
			// Create new tree and add to hashmap
			dtree = {
				id: treeId,
				nodeMap: new Map<number, DTreeNode>()
			}
			dTreeMap.set(treeId, dtree);
		}

		let node: DTreeNode;
		let nodeId = parseInt(row.node_id);

		const isLeaf = (row.is_leave.toLowerCase() == "true");
		//const nrSamples = parseInt(row.samples);
		//const impurity = parseFloat(row.impurity);
		const splitFeature = row.split_feature;
		const threshold = parseFloat(row.threshold);
		const correctPredictions = parseCorrectPredictions(row.correctPredictionsClassesCounts);
		const wrongPredictions = parseWrongPredictions(row.wrongPredictions);
		const correctPredictionCount = parseInt(row.correctPredictions)

		const allPredictions = mergePredictions(classCount, correctPredictions, wrongPredictions);

		if (nodeId == 0) {
			// Root node of the tree, create it
			node = {
				id: nodeId,
				depth: 0,
				isLeaf: isLeaf,
				//nrSamples: nrSamples,
				//impurity: impurity,
				splitFeature: splitFeature,
				threshold: threshold,
				children: [],
				allPredictions: allPredictions
			}

			dtree.rootNode = node;
		} else {
			// Node is a child and already exists
			node = {
				...dtree.nodeMap.get(nodeId) as DTreeNode,

				isLeaf: isLeaf,
				//nrSamples: nrSamples,
				//impurity: impurity,
				splitFeature: splitFeature,
				threshold: threshold,
				children: [],
				correctPredictions: correctPredictionCount,
				wrongPredictions: wrongPredictions,
				allPredictions: allPredictions
			}
		}

		if (!node.isLeaf) {
			node.children = row.children.split(";").map(x => parseInt(x));
			// Create dummy nodes for the children
			for (let childId of node.children) {
				let childNode = {
					id: childId,
					depth: node.depth + 1,
					isLeaf: false,
					children: [],
					allPredictions: { count: 0 }
				} as DTreeNode
				dtree.nodeMap.set(childNode.id, childNode)
			}
		}
		dtree.nodeMap.set(nodeId, node);
	}
	return dTreeMap;
}

function parseCorrectPredictions(correctPredictions: String): Predictions {
	//for example "2;11;1;9;0;10 means class2 11 samples, class1 9 samples, class0 10 samples 
	let predictions: Predictions = {
		count: 0
	}

	//no data
	if (correctPredictions == null || correctPredictions.indexOf(";") === -1) {
		return { "count": 0 }
	}

	let splits = correctPredictions.split(";"); // Split the input string by semicolon

	for (let i = 0; i < splits.length; i += 2) {
		let key = parseInt(splits[i]);
		let value = parseInt(splits[i + 1]);
		predictions[key] = { count: 0 }
		predictions[key][key] = value
	}


	return predictions;
}


// wrongPredictions: TotalCount;Original Class;MisclassifiedAs;Count;...;MisclassifiedAs;Count
function parseWrongPredictions(wrongPredictions: String): Predictions {
	// no wrong predictions
	if (wrongPredictions == null || wrongPredictions.indexOf(";") === -1) {
		return { "count": 0 }
	}
	let predObj = { "count": 0 };
	let splits = wrongPredictions.split(";"); // Split the input string by semicolon

	predObj["count"] = parseInt(splits[0]);
	for (let i = 1; i < splits.length; i += 3) {
		let key = splits[i];
		let subKey = splits[i + 1];
		let value = splits[i + 2]; //@ts-ignore
		predObj[key] = {}; //@ts-ignore 
		predObj[key][subKey] = parseInt(value);
	}

	return predObj
}

function mergePredictions(classCount: number, correctPredictions: Predictions, wrongPredictions: Predictions): Predictions {

	let allPredictions: Predictions = { count: 0 }//initialize all
	for (let originalClassI = 0; originalClassI < classCount; originalClassI++) {
		allPredictions[originalClassI] = { count: 0 } //initialize for this class
		for (let targetClassI = 0; targetClassI < classCount; targetClassI++) {
			let predictionsFrom = wrongPredictions;

			if (originalClassI == targetClassI) {
				predictionsFrom = correctPredictions; //these come from correct predictions
			}
			let count = 0;
			if (predictionsFrom[originalClassI] != undefined && predictionsFrom[originalClassI][targetClassI] != undefined) {
				count = predictionsFrom[originalClassI][targetClassI];
			}

			allPredictions[originalClassI][targetClassI] = count;
			allPredictions[originalClassI].count += count;
			allPredictions.count += count;
		}
	}
	return allPredictions;
}

export function readData(jsonData: any): Tree[] {
	let trees: Tree[] = [];
	for (let treeIndex in jsonData) {
		let jTree = jsonData[treeIndex];
		let tree = getTree(jTree);
		trees.push(tree);
	}
	normalizeData(trees);

	return trees;

}

export function readCluster(jsonData: any): Cluster {

	let convertedCentroids: Map<number, number> = new Map();

	for (let key of Object.keys(jsonData.centroids)) {
		convertedCentroids.set(+key, +jsonData.centroids[key])
	}

	let cluster = <Cluster>({
		centroids: convertedCentroids,
		labels: jsonData.labels,
		//linkage_Data: jsonData.linkage_Data,
		n_cluster: jsonData.n_cluster
	});

	return cluster;
}


function normalizeData(trees: Tree[]) {
	//get all the ranges in the correct order.
	for (let tree of trees) {
		for (let leafNode of tree.leafNodes) {
			let features = leafNode.normalizedRangesPerFeature;
			for (let i = 0; i < features.length; i++) {//go through the features one by one
				if (features[i][0] > features[i][1]) { //range is the wrong way around, fix it.
					let minVal = features[i][1];
					let maxVal = features[i][0];
					features[i][0] = minVal;
					features[i][1] = maxVal;
				}
			}
		}
	}

	//normalize data
	let minValues: number[] = [];
	let maxValues: number[] = []

	for (let tree of trees) {
		for (let leafNode of tree.leafNodes) {
			let features = leafNode.normalizedRangesPerFeature;
			for (let i = 0; i < features.length; i++) {//go through the features one by one
				if (Number.isNaN(minValues[i])) {
					minValues[i] = features[i][0];
				}
				if (Number.isNaN(maxValues[i])) {
					maxValues[i] = features[i][1];
				}

				minValues[i] = Math.min(minValues[i], features[i][0]); //first is always the lowest
				maxValues[i] = Math.max(maxValues[i], features[i][1]);
			}
		}
	}



	//normalize per feature
	for (let tree of trees) {
		for (let leafNode of tree.leafNodes) {
			let features = leafNode.normalizedRangesPerFeature;
			for (let i = 0; i < features.length; i++) {//go through the features one by one

				let range = maxValues[i] - minValues[i];
				let startOfRange = minValues[i];

				features[i][0] = (features[i][0] - startOfRange) / range;
				features[i][1] = (features[i][1] - startOfRange) / range;
			}
		}
	}

}

function getTree(jsonTree: any): Tree {
	let leafNodes: LeafNode[] = [];
	for (let jLeafNode of jsonTree["leafNodes"]) {
		// added this because otherwise after using slider class looks strange e.g. classclass0 instead of class0. 
		// maybe this can be fixed if readData is not called multiple times for the same rule data
		if (String(jLeafNode["class"]).indexOf("class") === -1) {
			jLeafNode["class"] = "" + jLeafNode["class"]; //make sure class is a string
		}
		leafNodes.push(jLeafNode);
	}
	let id = jsonTree["id"];
	let clusterDistances = jsonTree["rule_interval_sim"]

	let tree = <Tree>({
		leafNodes: leafNodes,
		id: id,
		clusterDistances: clusterDistances
	});
	return tree;
}
