import { Cluster, DTree, DTreeNode, FeatureMetaData, FeaturePlotData, Predictions, Rule } from "../types";
import * as PIXI from 'pixi.js';
import { featureColors } from "../colors";
import { focusFeature, selectCluster, unfocusFeature } from "../Interactions";
import { selectClusterProjection } from "../Projection";

import * as d3 from "d3";
import { printDuration, printStartTime } from "../Timing";

const horizontalMargin = 10;
const verticalMargin = 10;


let featureMap: Map<string, number> = new Map();

let originalValueScales: d3.ScaleLinear<number, number>[] = [];

/**
 * 
 * @param dTreeData 
 * @param cluster 
 * @param maxWidth 
 * @param features
 * @returns returns a map with representativeTreeId and the container visualizing it.
 */
export function generateFeaturePlots(dTreeData: Map<number, DTree>, cluster: Cluster, maxWidth: number, features: FeatureMetaData): Map<number, PIXI.Container> {
	printStartTime("calculating featureplot")

	//empty for new dataset
	featurePlotRuleDimRectangleOriginalSize = new Map<PIXI.Graphics, [number, number, number, number]>();
	dimRectangles = new Map<number, PIXI.Graphics[]>();
	ruleDimRectangles = new Map<number, PIXI.Graphics[]>();
	ruleDataMapping = new Map<PIXI.Graphics, Rule[]>();
	featureMap = new Map();
	originalValueScales = [];

	storeFeatureRanges(features);

	let featurePlotData = computeFeaturePlotData(dTreeData);
	let mergedFeatureData = mergeFeaturePlotData(featurePlotData, cluster)



	//generate a container holding the featureplot of each
	let containerMap = new Map();
	printDuration("calculating featureplot")
	printStartTime("draw feature plot")
	mergedFeatureData.forEach((data, id) => {
		let container = generateFeaturePlot(data, features);
		containerMap.set(id, container);
	})

	//set position of containers.
	let currentX = 0;
	let currentY = 0;
	containerMap.forEach(container => {
		let newWidth = container.getBounds().width;
		if ((currentX + newWidth + horizontalMargin) > maxWidth) { //Does not fit horizontally, take next ro
			currentX = 0;
			currentY += container.getBounds().height + verticalMargin;
		}
		container.x = currentX;
		container.y = currentY;
		currentX += newWidth + horizontalMargin;
	});

	printDuration("draw feature plot")
	return containerMap;
}

function storeFeatureRanges(features: FeatureMetaData) {
	for (let [featureName, feature] of Object.entries(features)) {
		let index = feature["index"];
		featureMap.set(featureName, index);

		let range = feature["range"];
		if (range === undefined) {
			let categories = feature["categories_norminal"];
			let maxCategoricalRange = categories[categories.length - 1];
			range = [0, maxCategoricalRange];
		}

		let originalValueScale = d3.scaleLinear()
			.range(range)
			.clamp(true) //Prevents negative values due to rounding when inverting the domain.
			.domain([0, 1]);

		originalValueScales[index] = originalValueScale;
	}
}


/**
 * @param featurePlotData 
 * @param cluster 
 * @returns <RepresentativeTreeid, combinedFeaturePlotData>
 */
function mergeFeaturePlotData(featurePlotData: Map<number, FeaturePlotData>, cluster: Cluster): Map<number, FeaturePlotData> {
	let clusterMapping = getFeaturesPlotDataToMerge(cluster);

	let mergedMapping = new Map();
	for (let representativeId of clusterMapping.keys()) {
		let representative = featurePlotData.get(representativeId)!
		//Merge all the features
		//Get all the levelfeatures data
		let dataArray: Map<number, [string, number][]>[] = [];
		dataArray.push(representative.levelFeatures);

		for (let representedId of clusterMapping.get(representativeId)!) {
			dataArray.push(featurePlotData.get(representedId)!.levelFeatures);
		}

		//Add up all those in merged
		let featuresPerLevel = new Map<number, number>();
		let featureData = new Map<number, [string, number][]>();
		for (let featurePlotDataX of dataArray) {
			for (let [level, featureLevelCount] of featurePlotDataX) {
				for (let [featureName, featureCount] of featureLevelCount) {
					//Increase the count for this level and this feature. If no array exists yet, make one
					featureData.set(level, increaseFeatureCount(featureData.get(level) || [], featureName, featureCount));
					featuresPerLevel.set(level, (featuresPerLevel.get(level) || 0) + featureCount);
				}
			}
		}

		//merge the rules
		let mergedRuleData: Map<number, Map<string, Rule[]>> = representative.featureLevelRules;

		for (let representedId of clusterMapping.get(representativeId)!) {
			let representeeData = featurePlotData.get(representedId)!.featureLevelRules;
			for (let [level, featureRule] of representeeData.entries()) {
				if (!mergedRuleData.has(level)) { mergedRuleData.set(level, new Map()) };//add empty if it doesn't exist
				let levelData = mergedRuleData.get(level)!;//Reference to object, so can simply edit

				for (let [featureName, rule] of featureRule.entries()) {
					if (!levelData.has(featureName)) { levelData.set(featureName, []) };//add empty if it doesn't exist
					let rules = levelData.get(featureName)!; //Reference to object, so can simply edit
					rules.push(...rule);//add the rules of the represented tree
				}
			}
		}

		let mergedData: FeaturePlotData = {
			id: representativeId,
			levelFeatures: featureData,
			levelTotalFeatures: featuresPerLevel,
			featureLevelRules: mergedRuleData
		};
		mergedMapping.set(representativeId, mergedData);
	}
	return mergedMapping;
}

function getFeaturesPlotDataToMerge(cluster: Cluster): Map<number, number[]> {
	let clusterMapping = new Map<number, number[]>();
	for (let [key, centroidId] of cluster.centroids) {
		let representedByCentroid: number[] = [];
		for (let i = 0; i < cluster.labels.length; i++) {
			if (cluster.labels[i] === +key) {
				representedByCentroid.push(i);
			}
		}
		clusterMapping.set(centroidId, representedByCentroid);
	}



	return clusterMapping;
}

/**For the feature with the specificed index, holds the dimrectangles per level for this feature */
function generateFeaturePlot(featurePlotData: FeaturePlotData, features: any): PIXI.Container {
	const PLOTWIDTH = 75.0 * 2;
	const PLOTHEIGHT = 75.0 * 2;

	let container = new PIXI.Container();
	let plot = new PIXI.Graphics();
	plot.interactive = true;
	container.addChild(plot);

	let totalHeight: number = PLOTHEIGHT;
	let heightEachLevel: number = totalHeight / featurePlotData.levelFeatures.size;

	let yPos: number = 0;
	// For each level draw the feature rects
	for (let level = 0; level < featurePlotData.levelFeatures.size; level++) {

		const featuresCount: [string, number][] = featurePlotData.levelFeatures.get(level)!;
		featuresCount.sort((a, b) => featureMap.get(a[0])! - featureMap.get(b[0])!)

		const normalizationFactor = 1 / featuresCount.reduce((accumulator, item) => accumulator + item[1], 0);
		let xPos: number = 0;
		for (let i = 0; i < featuresCount.length; i++) {
			let featureIndex = featureMap.get(featuresCount[i][0])!;

			let rectWidth = featuresCount[i][1] * normalizationFactor * PLOTWIDTH;

			let color = new PIXI.Color(featureColors(featureMap.get(featuresCount[i][0])! / Object.keys(features).length));
			color.setAlpha(0.75);
			plot.rect(xPos, yPos, rectWidth, heightEachLevel);
			plot.fill(color);
			plot.stroke({ width: 1, color: 0x000000, alpha: 0.2 });


			//add the dimming interaction
			let ruleData = featurePlotData.featureLevelRules.get(level)!.get(featuresCount[i][0])!;
			addDimRectangle(container, ruleData, featureIndex, xPos, yPos, rectWidth, heightEachLevel, featurePlotData.id);

			xPos += rectWidth;
		}

		yPos += heightEachLevel;
	}

	addClickInteraction(plot, featurePlotData.id);

	return container;
};

/**Holds the original [xpos,rectwidth] of each dimRectangle */
let featurePlotRuleDimRectangleOriginalSize = new Map<PIXI.Graphics, [number, number, number, number]>();
let dimRectangles = new Map<number, PIXI.Graphics[]>();
let ruleDimRectangles = new Map<number, PIXI.Graphics[]>();
let ruleDataMapping = new Map<PIXI.Graphics, Rule[]>();
function addDimRectangle(container: PIXI.Container, ruleData: Rule[], featureNumber: number, xPos: number, yPos: number, rectWidth: number, heightEachLevel: number, treeIndex: number) {
	//Need to add two types of dimming rectangles, one for the rules, one for the features.
	//Needed as we can have partial dimming and partial filtering
	let ruleDimRectangle = new PIXI.Graphics();
	ruleDimRectangle.rect(xPos, yPos, rectWidth, heightEachLevel);
	ruleDimRectangle.fill(0xffffff);
	ruleDimRectangle.visible = false;
	container.addChild(ruleDimRectangle);


	featurePlotRuleDimRectangleOriginalSize.set(ruleDimRectangle, [xPos, yPos, rectWidth, heightEachLevel]);
	ruleDataMapping.set(ruleDimRectangle, ruleData);
	if (!ruleDimRectangles.has(featureNumber)) { ruleDimRectangles.set(featureNumber, []) };
	ruleDimRectangles.get(featureNumber)!.push(ruleDimRectangle);


	let dimRectangle = new PIXI.Graphics();

	dimRectangle.rect(xPos, yPos, rectWidth, heightEachLevel);
	dimRectangle.fill(0xffffff)
	dimRectangle.alpha = 0; //start invisible
	dimRectangle.interactive = true; //only need to have interaction for the top one
	dimRectangle.hitArea = new PIXI.Rectangle(xPos, yPos, rectWidth, heightEachLevel);
	container.addChild(dimRectangle);

	dimRectangle.onmouseover = function (mouseData) {
		focusFeature(featureNumber);
		selectClusterProjection(treeIndex);
	}
	dimRectangle.onmouseleave = function (mouseData) {
		unfocusFeature(featureNumber);
		selectClusterProjection();
	}

	if (!dimRectangles.has(featureNumber)) { dimRectangles.set(featureNumber, []) };
	dimRectangles.get(featureNumber)!.push(dimRectangle);
}

/**Undims a specific feature in the featureplot */
export function undimFeaturePlotFeature(featureIndex: number) {
	for (let rectangle of dimRectangles.get(featureIndex)!) {
		rectangle.alpha = 0;
	}
}

/**Dims a specific feature in the featureplot */
export function dimFeaturePlotFeature(featureIndex: number) {
	for (let rectangle of dimRectangles.get(featureIndex)!) {
		rectangle.alpha = 0.6
	}
}

export function dimFeaturePlot(featureRanges: [number, number][], classes: boolean[][]) {

	for (let featureIndex = 0; featureIndex < featureRanges.length; featureIndex++) {
		for (let rectangle of ruleDimRectangles.get(featureIndex)!) {
			let rules = ruleDataMapping.get(rectangle)!;
			let fractionOutsideRange = getFractionOutsideRange(featureRanges, classes, rules);
			let fractionInRange = 1 - fractionOutsideRange;
			let [originalX, originalY, originalWidth, originalHeight] = featurePlotRuleDimRectangleOriginalSize.get(rectangle)!;

			let targetWidth = fractionOutsideRange * originalWidth;
			if (targetWidth < 0.01) { //can't draw tiny rectangles. Results in artifacts
				rectangle.visible = false;
				continue;
			}

			rectangle.alpha = 0.9;
			rectangle.visible = true;
			rectangle.x = originalX * (fractionInRange) + originalWidth * fractionInRange; //This is not going correct. There is scaling going on horizontally, this needs to be resolved.
			rectangle.width = targetWidth;
		}
	}
}

/**
 * 
 * @param testRanges 
 * @param dataRangesArray 
 * @returns A values between 0 and 1
 */
function getFractionOutsideRange(targetRanges: [number, number][], classificationsToShow: boolean[][], rules: Rule[]): number {

	let normalizedRules: Rule[] = [];
	for (let rule of rules) {
		let normalizedRange = getNormalizedRule(rule.ranges)
		let normalizedRule: Rule = { ranges: normalizedRange, predictions: rule.predictions };
		normalizedRules.push(normalizedRule);
	}
	return 1 - getPercentageOfValidRules(normalizedRules, targetRanges, classificationsToShow);
}

function allTrue(array: boolean[][]): boolean {
	for (let i = 0; i < array.length; i++) {
		for (let j = 0; j < array.length; j++) {
			if (array[i][j] == false) {
				return false;
			}
		}
	}
	return true;
}

function computeFeaturePlotData(dTreeMap: Map<number, DTree>): Map<number, FeaturePlotData> {

	let featurePlotDataMap = new Map<number, FeaturePlotData>();

	for (const treeId of dTreeMap.keys()) {
		let dTree = dTreeMap.get(treeId)!;

		let featurePlotData: FeaturePlotData = {
			id: dTree.id,
			levelFeatures: new Map(),
			levelTotalFeatures: new Map(),
			featureLevelRules: new Map()
		};

		let initialRule: Rule = { ranges: [], predictions: { count: 0 } };
		for (let i = 0; i < featureMap.size; i++) {
			initialRule.ranges[i] = [NaN, NaN]
		}

		processNode(dTree, dTree.rootNode!, initialRule, featurePlotData);
		featurePlotDataMap.set(treeId, featurePlotData)
	}

	return featurePlotDataMap;
}


/**
 * For adds all rules this is involved in.
 */
function processNode(dTree: DTree, currentNode: DTreeNode, currentRule: Rule, featurePlotData: FeaturePlotData): Rule[] {
	let featureIndex = featureMap.get(currentNode.splitFeature!)!;
	let splitFeature = currentNode.splitFeature!;
	let splitPoint = currentNode.threshold!;
	let depth = currentNode.depth;

	// Update levelTotalFeatures
	let nrFeatures: number = featurePlotData.levelTotalFeatures.get(depth) ?? 0; //0 as default
	featurePlotData.levelTotalFeatures.set(depth, nrFeatures + 1);

	//update levelFeatures
	let featuresForLevel = featurePlotData.levelFeatures.get(depth) ?? [];
	featuresForLevel = increaseFeatureCount(featuresForLevel, currentNode.splitFeature!, 1);
	featurePlotData.levelFeatures.set(depth, featuresForLevel);

	let levelRule = featurePlotData.featureLevelRules.get(depth) ?? new Map<string, Rule[]>();
	let rules = levelRule.get(splitFeature) ?? [];
	//make sure the references are initialized
	if (levelRule.size === 0) {
		featurePlotData.featureLevelRules?.set(depth, levelRule);
	}
	if (rules.length === 0) {
		levelRule.set(splitFeature, rules);
	}

	//Recurse and update rules

	const leftChild = dTree.nodeMap.get(currentNode.children[0])!
	const rightChild = dTree.nodeMap.get(currentNode.children[1])!

	let leafRules: Rule[] = [];

	//handle left child
	//Update the rule for the left child, don't modify the existing rule so we can split both ways.
	let leftNewRule = structuredClone(currentRule);
	leftNewRule.ranges[featureIndex][1] = splitPoint; //Left is always smaller in the decision tree
	if (!leftChild.isLeaf) {
		//recurse with the updated rule and store the result in featurePlotData.
		let leftRules = processNode(dTree, leftChild, leftNewRule, featurePlotData);
		leafRules.push(...leftRules);
	} else {
		//no left recursion. return the completed rule
		leftNewRule.predictions = leftChild.allPredictions;
		leafRules.push(leftNewRule);
	}

	//do the same for the right child
	let rightNewRule = structuredClone(currentRule);

	rightNewRule.ranges[featureIndex][0] = splitPoint; //right is always larger in the decision tree
	if (!rightChild.isLeaf) {
		//recurse with the updated rule and store the result in featurePlotData.
		let rightRules = processNode(dTree, rightChild, rightNewRule, featurePlotData);
		leafRules.push(...rightRules);
	} else {
		//no right recursion. return the completed rule
		rightNewRule.predictions = rightChild.allPredictions;
		leafRules.push(rightNewRule);
	}
	rules.push(...leafRules);
	return leafRules;
}


/**
 * Finds the feature with {featureName} in {featureCount} and adds {increase} to the count, or sets it to {increaseCountBy} if it did not exist yet.
 * @param featureCount The list to be added, if undefined it will be initilaized
 * @param featureName 
 * @param increase By how much we are increasing this value
 * @returns 
 */
function increaseFeatureCount(featureCount: [string, number][] | undefined, featureName: string, increaseCountBy: number): [string, number][] {
	if (featureCount == undefined) {
		featureCount = []
	}
	//try and increase the count of the feature
	let isInList: boolean = false
	for (let i = 0; i < featureCount.length; i++) {
		if (featureCount[i][0] == featureName) {
			// feature is in list, increase count
			featureCount[i][1] = featureCount[i][1] + increaseCountBy;
			isInList = true;
			break;
		}
	}
	// If current feature is not in the list, add it and increase count
	if (!isInList) {
		featureCount.push([featureName, increaseCountBy])
	}

	return featureCount
}



/**
 * For the array of tested rules, returns the percentage valid features
 * @param normalizedTestRules 
 * @param targetRanges 
 */
function getPercentageOfValidRules(normalizedTestRules: Rule[], targetRanges: [number, number][], classificationsToShow: boolean[][]): number {

	let validRules = 0;

	for (let rule of normalizedTestRules) {
		if (isWithinRange(rule.ranges, targetRanges) && inSelectedClasses(rule.predictions, classificationsToShow)) {
			validRules++;
		}
	}
	let percentageValid = validRules / normalizedTestRules.length;
	return percentageValid;
}

/**
 * Returns whether predictions contains at least one of the specified classes
 * @param predictions 
 * @param classes 
 * @returns 
 */
function inSelectedClasses(predictions: Predictions, classes: boolean[][]): boolean {
	for (let classI = 0; classI < classes.length; classI++) {
		for (let classJ = 0; classJ < classes.length; classJ++) {
			//class wasn't selected
			if (!classes[classI][classJ]) {
				continue;
			}
			//at least 1 prediction
			if (predictions[classI][classJ] > 0) {
				return true;
			}
		}
	}

	return false;
}


/**
 * Returns whether testRange is within targetRange for all features.
 * 
 * @param testRanges 
 * @param targetRanges 
 */
function isWithinRange(testRanges: [number, number][], targetRanges: [number, number][]): boolean {
	if (testRanges.length != targetRanges.length) {
		console.error(`testranges ${testRanges.length} does not have the same length as targetranges ${targetRanges.length}`)
	}
	for (let i = 0; i < testRanges.length; i++) {
		let testRange = testRanges[i];
		let targetRange = targetRanges[i];
		//if at least 1 feature does not overlap, it is not within range
		if (testRange[0] > targetRange[1]) { //testRange starts after target ends
			return false;
		}
		if (testRange[1] < targetRange[0]) { //testRange ends before target starts
			return false;
		}
	}
	return true;
}

function addClickInteraction(plot: any, treeId: number) {
	plot.onclick = function () {
		// select cluster for node link
		selectCluster(treeId);
	}
}

function getNormalizedRule(rule: [number, number][]): [number, number][] {
	//a bit inefficient with how often this is called, can also store the normalized values earlier if needed.
	let normalizedRule: [number, number][] = [];
	for (let i = 0; i < rule.length; i++) {
		normalizedRule[i] = [originalValueScales[i].invert(rule[i][0]), originalValueScales[i].invert(rule[i][1])];

		if (Number.isNaN(normalizedRule[i][0])) {
			normalizedRule[i][0] = 0;
		}
		if (Number.isNaN(normalizedRule[i][1])) {
			normalizedRule[i][1] = 1;
		}
	}
	return normalizedRule;
}
