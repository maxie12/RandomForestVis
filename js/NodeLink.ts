import * as d3 from "d3";
import { flextree } from 'd3-flextree'; //@ts-ignore
import { focusFeature, unfocusFeature } from "./Interactions";
import { classColors, featureColors } from "./colors";
import { kernelEpanechnikov, kernelDensityEstimator, silvermansRule } from "./Kernels";
import { Cluster, DecisionTreeNode, Classes, Predictions } from "./types";
import { getMaxFontSize } from "./TextUtils";
import { printDuration, printStartTime } from "./Timing";

let globalTrees: DecisionTreeNode[];
let globalFeatures: any;
let globalClasses: Classes;
let globalClusters: Cluster;

const featureMap: Map<string, number> = new Map();
const classMap: Map<string, number> = new Map();
const colorScales: any = {};
const yAxes: any = {};
const kdeValues: any = {};
const originalScales: any = {};
const numberOfCategories: any = {};

const grayColor = "#d6dfe3";
const histograms: any = {};
let x: d3.ScaleLinear<number, number, never>;

//Target widths
const desiredTotalWidth = 600;//maximum width the biggest tree can take. 
let baseFontSize = 50;//Tries to use the specified font if it fits, can be smaller depending on amount of leafs.
let textPercentageMargin = 0.1;

//saves and stores all the widhts
let nodeWidth = NaN;
let nodeHeight = NaN;
let nodeMargin = [NaN, NaN];

let leafWidth = NaN;
let leafHeight = NaN;
let leafMargin = [NaN, NaN];

let edgeWidthRange = [NaN, NaN]
let fontSize = NaN;
let strokeScale = NaN;

/**
 * 
*   @param treeData
 * @param svg Svg element where the trees are being added on. Required to calculate the height of text elements
 */
function initTreeSizing(rootNodes: DecisionTreeNode[], classes: Classes, svg: d3.Selection<SVGGElement, unknown, HTMLElement, any>) {

    //get how many nodes of width we require for the largest tree.
    let horizontalSpaceCount = getHorizontalSpaceCount(rootNodes)

    const nodeMarginPercentage = 0.1;
    const nodePercentage = 1 - nodeMarginPercentage;

    nodeWidth = desiredTotalWidth / horizontalSpaceCount * nodePercentage;
    nodeMargin[0] = desiredTotalWidth / horizontalSpaceCount * nodeMarginPercentage
    leafWidth = nodeWidth;
    leafMargin[0] = nodeMargin[0]
    edgeWidthRange = [nodeWidth / 100 * 5, nodeWidth / 100 * 15]; //5% to 15%

    //Widths are calculated. Find out which font we have for the height
    let availableWidth = leafWidth * (1 - textPercentageMargin);
    let maxFontSize = getMaxFontSize(availableWidth, Object.keys(classes), svg)

    //Don't go beyond the default size  the golden ratio. Relevant for very short labels(glass dataset) or the default max font size (iris)
    fontSize = Math.min(maxFontSize, baseFontSize, nodeWidth / 1.161 / 2); //extra divide by 2 as that is the leaf height

    let textHeight = fontSize;//fontsize is equal to height of maximal character.
    leafHeight = textHeight * 2;
    leafMargin[1] = leafHeight //need quite a big margin to allow for the edges to draw nicely

    nodeHeight = leafHeight;
    nodeMargin[1] = nodeHeight //need quite a big margin to allow for the edges to draw nicely

    strokeScale = nodeWidth / 67.5; //67.5 is a IRIS default value
}

/**
 * 
 * @param rootNodes 
 * @returns //how many nodes of equal width the largest tree has.
 */
function getHorizontalSpaceCount(rootNodes: DecisionTreeNode[]) {
    //@ts-ignore
    const layout = flextree()
        .nodeSize([1, 1]);

    let maxWidth = 0;
    for (let tree of rootNodes) {
        const root = d3.hierarchy(tree);

        //@ts-ignore
        const treeLayout = layout(root);
        const nodes = treeLayout.descendants();
        let minX = Infinity;
        let maxX = -Infinity;
        for (let node of nodes) {
            minX = Math.min(minX, node.x)
            maxX = Math.max(maxX, node.x)
        }
        maxWidth = Math.max(maxWidth, maxX - minX);
    }
    return maxWidth;
}


// only show trees of the currently selected cluster, first show the representative tree
export function initializeTrees(treeData: DecisionTreeNode[], clusterData: Cluster, features: any, classes: Classes) {
    printStartTime("initialize trees");

    d3.select('#decisionTrees').select("svg").remove();

    addPredictionsToTreeData(treeData, classes);

    const svg = d3.select("#decisionTrees")
        .append("svg")
        .attr("overflow", "hidden")

    const g = svg.append("g").attr("transform", "translate(0,0) scale(1)"); //used for zooming. Set default so we can get it in updatePositions
    initTreeSizing(treeData, classes, g)

    x = d3.scaleLinear()
        .domain([0, 1])
        .range([0, nodeWidth]);

    globalTrees = treeData;
    globalFeatures = features;
    globalClasses = classes;
    globalClusters = clusterData;
    for (let feature of Object.keys(features)) {
        featureMap.set(feature, features[feature]["index"]);
        if (features[feature]["range"] === undefined) {
            histograms[feature] = computeHistogram(features[feature]["numerical_categories"], features[feature]["scaled_values"], x, nodeHeight, feature)
            originalScales[feature] = d3.scaleLinear().domain(features[feature]["categories_norminal"]).range(features[feature]["numerical_categories"]);
            numberOfCategories[feature] = features[feature]["categories"].length
        } else {
            kdeValues[feature] = computeKDE(features[feature]["scaled_values"], x, nodeHeight, feature);
            originalScales[feature] = d3.scaleLinear().domain(features[feature]["range"]).range([0, 1]);
        }
    }


    for (let cl of Object.keys(classes)) {
        classMap.set(cl, classes[cl]["index"]);
        let col = classColors(classes[cl]["index"] / Object.keys(globalClasses).length);
        //@ts-ignore
        colorScales[cl] = d3.scaleLinear().domain([1 / Object.keys(globalClasses).length, 1]).range([grayColor, col]);
    }
    // use the representative of cluster 1 as default
    const clusterNumber = 1;
    const treeIndex = clusterData["centroids"].get(clusterNumber)!;
    const otherTrees = clusterData["labels"].reduce((acc: any, val: number, index: number) => {
        //skip rep tree  
        if (index === treeIndex) {
            return acc
        }
        if (val === clusterNumber) {
            acc.push(index);
        }
        return acc;
    }, []);

    printDuration("initialize trees");
    printStartTime("draw trees");
    createTree(treeData[treeIndex], treeIndex, true);

    for (let index of otherTrees) {
        createTree(treeData[index], index);
    }
    let [totalWidth, totalHeight] = positionDecisionTrees();

    svg.attr("width", totalWidth);
    svg.attr("height", totalHeight);

    const zoom = d3.zoom()
        .extent([[0, 0], [totalWidth, totalHeight]])
        .scaleExtent([0.125, 20])
        // .translateExtent([[0,0],[totalWidth,totalHeight]])
        .filter(scrollDisable) //prevent scrolling
        .on('zoom', zoomed);

    //@ts-ignore
    function zoomed({ transform }) {
        if (transform.x > 0) {
            transform.x = 0;
        }
        if (transform.x / transform.k < -totalWidth) {
            transform.x = -totalWidth;
        }
        if (transform.y > 0) {
            transform.y = 0;
        }
        if (transform.y / transform.k < -totalHeight) {
            transform.y = -totalHeight;
        }
        g.attr("transform", transform);
    }

    //@ts-ignore
    svg.call(zoom);
    //@ts-ignore
    zoom.scaleBy(svg,0.6);
    //prevent scrolling
    //@ts-ignore
    function scrollDisable(event) {
        event.preventDefault();
        return (!event.ctrlKey || event.type === 'wheel') && !event.button;
    }
    printDuration("draw trees");
}

function createTree(tree: DecisionTreeNode, treeNumber: number, isCentroid: boolean = false): d3.Selection<SVGSVGElement, unknown, HTMLElement, any> {
    const svgToAddTo = d3.select("#decisionTrees")
        .select("svg").select("g");

    if (svgToAddTo.node() == null) {
        throw new Error("#decisionTrees does not have an svg element or does not exist");
    }

    const svg = svgToAddTo
        .append("svg")
        .attr("class", "decisionTree")

    //generate the layout
    const root = d3.hierarchy(tree);

    //@ts-ignore
    const layout = flextree()
        .nodeSize((node: any) => {
            node.size = [nodeWidth, nodeHeight];//size of nodes to draw
            return [nodeWidth + nodeMargin[0], nodeHeight + nodeMargin[1]] //size of space to reserve
        })

    //@ts-ignore
    const treeLayout = layout(root);

    let maxSamples = valueSum(root.data.values);
    let edgeWidth = d3.scaleLinear()
        .domain([0, maxSamples])
        .range(edgeWidthRange);


    // Find leftmostnode for offsetting the svg
    const nodes = treeLayout.descendants();
    const leftMostNode = nodes.reduce((minNode, node) => (node.x < minNode.x ? node : minNode), nodes[0]);
    const offsetX = Math.abs(leftMostNode.x) + leftMostNode.size[0] / 2;


    const offsetY = 30 //offset by 30 to make space for text
    const g = svg.append("g").attr("transform", `translate(${offsetX}, ${offsetY + nodeHeight})`)
    const accuracy = " Accuracy: " + tree["accuracy"].toFixed(3)
    const treeText = (isCentroid ? "Central tree of cluster " + treeNumber : "Tree " + treeNumber) + "." + accuracy

    if (isCentroid) {
        const treeHeading = document.getElementById("tree-heading");
        if (treeHeading) {
            treeHeading.textContent = `Decision trees within cluster ${treeNumber}`;
        }
    }

    //position text above first node
    g.append("text")
        .attr("text-anchor", "left")
        .attr("font-size", "25px")
        .attr("transform", `translate(${leftMostNode.x - nodeWidth / 2}, ${-nodeHeight})`)
        .text(treeText);

    // Draw links
    g.selectAll(".link")
        .data(root.links())
        .enter()
        .append("path")
        .attr("class", "link")
        .attr("d", (d) => {
            const targetId: number = d.target.data.node_id;
            const index = d.source.children!.findIndex((child) => child.data.node_id === targetId);//1 is left, 2 is right
            const sign = index == 1 ? 1 : -1;
            let widthE = edgeWidth(valueSum(d.target.data.values)) / 2
            let xOffset = widthE * sign;
            let feature = d.source.data.feature
            let threshold = d.source.data.threshold

            //@ts-ignore
            let startX = d.source.x - d.source.size[0] / 2;
            //@ts-ignore
            let thresholdX = startX + originalScales[feature](threshold) * d.source.size[0]

            //@ts-ignore
            const sourceOffset = d.source.size[1] / 2
            //@ts-ignore
            const targetOffset = d.target.size[1] / 2;
            //@ts-ignore
            return `M${thresholdX + xOffset},${d.source.y + sourceOffset} C${thresholdX + xOffset},${(d.source.y + d.target.y) / 2} ${d.target.x},${(d.source.y + d.target.y) / 2} ${d.target.x},${d.target.y - targetOffset}`

        })
        .style("fill", "none")
        .style("stroke", function (d) {
            return blendColors(d.target.data.values)
        })
        .style("stroke-width", function (d) {
            let flowValue = valueSum(d.target.data.values);
            return edgeWidth(flowValue)
        })
        .append("title").text(d => valueSum(d.target.data.values))


    // Draw nodes
    const node = g.selectAll(".node")
        .data(root.descendants())
        .enter()
        .append("g")
        .attr("class", "node")
        .attr("transform", (d: any) => `translate(${d.x}, ${d.y})`)
        .on("mouseover", function (event, d) {
            if (d.data.feature === "Prediction") {
                return
            }
            let feature = d.data.feature;
            let featureNumber = featureMap.get(feature)!;
            focusFeature(featureNumber);

        })
        .on("mouseleave", (event, d) => {
            let feature = d.data.feature;
            let featureNumber = featureMap.get(feature)!;
            unfocusFeature(featureNumber);
        })

    node.each(function (d) {
        let self = d3.select(this);
        if (d.data.children.length == 0) {
            // Draw rectangles for leaves
            drawLeaf(self)
        } else {
            // draw feature plot with split point
            let color = featureColors(featureMap.get(d.data.feature)! / Object.keys(globalFeatures).length);
            drawRect(self)
            if (!globalFeatures[d.data.feature].is_categorical) {
                drawKDE(self, d, color, x)
            } else {
                drawHistogram(self, d, color, x)
            }
        }
    })


    return svg;
}

//two rectanlges. the second one to show missclassification
function drawLeaf(node: any) {
    let color: d3.RGBColor;
    node.append("rect")
        .attr("class", (d: any) => "nodeFeature-" + featureMap.get(d.data.feature))
        .attr("x", (d: any) => -d.size[0] / 2)
        .attr("y", (d: any) => -d.size[1] / 2)
        .attr("width", (d: any) => d.size[0])
        .attr("height", (d: any) => d.size[1])
        //.attr("rx", 5)
        .attr("fill", function (d: d3.HierarchyNode<DecisionTreeNode>) {
            color = d3.rgb(classColors(classMap.get(d.data.threshold.toString())! / Object.keys(globalClasses).length))
            ////Manual interpolation of white with the he color
            // color = d3.rgb(color.r * 1.25, color.g * 1.25, color.b * 1.25)
            return color;
        })
        .attr("fill-opacity", 1)
        .attr("stroke-width", 1 * strokeScale)
        .attr("stroke-opacity", 1)
        .attr("stroke", "none");



    node.append("text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", (d: any) => {
            //https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
            //Use relative lightness to determine if black of white text should be used. Not perfectly implemented, but good enough
            if ((color.r * 0.299 + color.g * 0.587 + color.b * 0.114) > 186) {
                return "#000000"
            } else {
                return "#ffffff";
            }
        })
        .attr("stroke", "none")
        .attr("font-size", fontSize + "px")
        .text((d: d3.HierarchyNode<DecisionTreeNode>) => d.data.threshold)
    


    node.append("title").text((d: d3.HierarchyNode<DecisionTreeNode>) =>
        (d.data.children.length > 0 ? `${d.data.feature} <= ${d.data.threshold}` : `${d.data.feature}: ${d.data.threshold}`),);

    const dataPoint = node.data()[0]
    const totalPredictions = dataPoint.data.correctPredictions + dataPoint.data.wrongPredictions.count
    const widthScale = d3.scaleLinear().range([0, dataPoint.size[0]]).domain([0, totalPredictions])
    // not every leaf is visited by the test dataset
    if (dataPoint.data.correctPredictions > 0) {
        //correct classified
        node.append("rect")
            .attr("class", (d: d3.HierarchyNode<DecisionTreeNode>) => "nodeFeature-" + featureMap.get(d.data.feature))
            .attr("x", (d: any) => -d.size[0] / 2)
            .attr("y", (d: any) => (-d.size[1] / 2) + d.size[1])
            .attr("width", (d: any) => widthScale(d.data.correctPredictions))
            .attr("height", (d: any) => d.size[1] / 2)
            .attr("fill", function (d: d3.HierarchyNode<DecisionTreeNode>) {
                let color = d3.color(classColors(classMap.get(d.data.threshold.toString())! / Object.keys(globalClasses).length))
                return color;
            })
            .attr("fill-opacity", 1)
            .attr("stroke-width", 1 * strokeScale)
            .attr("stroke-opacity", 1)
            .attr("stroke", "none");
    }

    if (dataPoint.data.wrongPredictions.count > 0) {
        // Assuming dataPoint.data.wrongPredictions is an object
        let wrongPredictions = { ...dataPoint.data.wrongPredictions };
        // Remove the count attribute
        delete wrongPredictions.count;
        // Transform the object into an array of [key, value] pairs
        wrongPredictions = Object.entries(wrongPredictions);
        let currentX = widthScale(dataPoint.data.correctPredictions);

        wrongPredictions.forEach((d: any) => {
            let width = widthScale(d[1]["count"])
            node.append("rect")
                .attr("x", (-dataPoint.size[0] / 2) + currentX)
                .attr("y", (-dataPoint.size[1] / 2) + dataPoint.size[1])
                .attr("width", width)
                .attr("height", dataPoint.size[1] / 2)
                //.attr("rx", 5)
                .attr("fill", function () {
                    let color = d3.color(classColors(Number(d[0]) / Object.keys(globalClasses).length))
                    return color;
                })
                .attr("fill-opacity", 1)
                .attr("stroke-width", 1 * strokeScale)
                .attr("stroke-opacity", 1)
                .attr("stroke", "none");

            currentX += width;
        });
    }

    //not visited by the test set
    if (totalPredictions === 0) {
        node.append("rect")
            .attr("class", (d: d3.HierarchyNode<DecisionTreeNode>) => "nodeFeature-" + featureMap.get(d.data.feature))
            .attr("x", (d: any) => -d.size[0] / 2)
            .attr("y", (d: any) => (-d.size[1] / 2) + d.size[1])
            .attr("width", dataPoint.size[0])
            .attr("height", (d: any) => d.size[1] / 2)
            .attr("fill", grayColor)
            .attr("fill-opacity", 1)
            .attr("stroke-width", 1 * strokeScale)
            .attr("stroke-opacity", 1)
            .attr("stroke", "none");
    }
}


function drawRect(node: any) {
    node.append("rect")
        .attr("class", (d: any) => "nodeFeature-" + featureMap.get(d.data.feature))
        .attr("x", (d: any) => -d.size[0] / 2)
        .attr("y", (d: any) => -d.size[1] / 2)
        .attr("width", (d: any) => d.size[0])
        .attr("height", (d: any) => d.size[1])
        .attr("rx", (d: any) => d.size[0] / 20)
        .attr("fill", "white")
        .attr("fill-opacity", 1)
        .attr("stroke-width", 1 * strokeScale)
        .attr("stroke-opacity", 1)
        .attr("stroke", function (d: d3.HierarchyNode<DecisionTreeNode>) {
            return featureColors(featureMap.get(d.data.feature)! / Object.keys(globalFeatures).length);
        });
    node.append("title").text((d: any) => `${d.data.feature} <= ${d.data.threshold}`);
}

function valueSum(values: number[]) {
    return values.reduce((a: number, b: number) => a + b, 0)
}

function blendColors(values: number[]) {
    let samples = valueSum(values)
    //there are no samples in the test dataset for this node
    if (samples === 0) {
        return grayColor
    }
    let maxIndex = values.indexOf(Math.max(...values));
    //@ts-ignore
    let className = Object.entries(globalClasses).find(x => x[1]["index"] == maxIndex)[0];
    let scale = colorScales[className];
    return scale(values[maxIndex] / samples)
}

function computeKDE(values: number[], x: any, size: number, feature: string) {
    const bandwidth = silvermansRule(values, 1);
    const kde = kernelDensityEstimator(kernelEpanechnikov(bandwidth), x.ticks(100));
    const kdeValues = kde(values);

    let yAxis = d3.scaleLinear()
        .range([size, 0])
        .domain([0, Math.max(...kdeValues.map((a: any) => a[1]))]);
    yAxes[feature] = yAxis;
    return kdeValues;
}

function drawKDE(target: any, dataPoint: any, color: string, x: any) {
    let feature = dataPoint.data.feature;
    let size = dataPoint.size; //@ts-ignore
    let scale = originalScales[feature]; //@ts-ignore
    let yScale = yAxes[feature];
    let ancestorFeature = dataPoint.data.ancestorFeatures[feature];
    // we only need grey in case of previous splitpoints of the same feature
    let needGreyArea = ((ancestorFeature && ancestorFeature.length >= 1) ? true : false)

    target.append("path")
        .attr("class", "feature-chart")
        .attr("fill", function () {
            if (needGreyArea) {
                return grayColor;
            }
            return color;
        })
        .datum(kdeValues[feature])
        .attr("opacity", ".8")
        .attr("stroke-linejoin", "round")
        .attr("transform", `translate( ${-size[0] / 2}, ${-size[1] / 2})`)
        .attr("d", d3.area()
            .curve(d3.curveBasis)
            .x(function (d: any) { return x(d[0]); })
            .y1(function (d: any) {
                return yScale(d[1]);
            })
            .y0(function () {
                return yScale(0);
            }))

    // colored overlay to show valid feature range
    if (needGreyArea) {
        target.append("path")
            .attr("class", "feature-chart")
            .attr("fill", color)
            .datum(function () {
                let values = kdeValues[feature];
                // get biggest ancestor with right orientation as start point -> otherwise use 0
                let rightFeature = ancestorFeature.reduce((max: any, item: any) => {
                    if (item.orientation === "right" && (!max || item.threshold > max.threshold)) {
                        return item;
                    }
                    return max;
                }, null);
                rightFeature = (rightFeature == null ? 0 : scale(rightFeature.threshold))
                // get smallest ancestor with left as end point -> otherwise use 1
                let leftFeature = ancestorFeature.reduce((min: any, item: any) => {
                    if (item.orientation === "left" && (!min || item.threshold < min.threshold)) {
                        return item;
                    }
                    return min;
                }, null);
                leftFeature = (leftFeature == null ? 1 : scale(leftFeature.threshold))
                values = values.filter((x: any) => x[0] >= rightFeature && x[0] <= leftFeature)
                return values;
            })
            .attr("opacity", ".8")
            .attr("stroke-linejoin", "round")
            .attr("transform", `translate( ${-size[0] / 2}, ${-size[1] / 2})`)
            .attr("d", d3.area()
                .curve(d3.curveBasis)
                .x(function (d: any) { return x(d[0]); })
                .y1(function (d: any) {
                    return yScale(d[1]);
                })
                .y0(function () {
                    return yScale(0);
                }))
    }

    target.append("line")
        .attr("x1", function () {
            return x(scale(dataPoint.data.threshold))
        })
        .attr("x2", function () {
            return x(scale(dataPoint.data.threshold))
        })
        .attr("y1", 0)
        .attr("y2", size[1])
        .style("stroke-width", 2 * strokeScale)
        .style("stroke", "black")
        .style("fill", "none")
        .attr("transform", `translate( ${-size[0] / 2}, ${-size[1] / 2})`);
}


function computeHistogram(values: any, scaled_vales: any, x: any, size: number, feature: string) {
    const histogram = d3.histogram()
        .value((v: any) => v)
        .domain(x.domain())
        .thresholds(values);
    const bins = histogram(scaled_vales);
    const yAxis = d3.scaleLinear()
        .range([0, size])
        .domain([0, Math.max(...bins.map((b: any) => b.length))]);
    yAxes[feature] = yAxis;
    return bins;
}

function drawHistogram(target: any, dataPoint: any, color: string, x: any) {
    let feature = dataPoint.data.feature;
    let size = dataPoint.size; //@ts-ignore
    let scale = originalScales[feature];
    let yScale = yAxes[feature];
    let ancestorFeature = dataPoint.data.ancestorFeatures[feature];
    // we only need grey in case of previous splitpoints of the same feature
    let needGreyArea = ((ancestorFeature && ancestorFeature.length >= 1) ? true : false)
    let barWidth = (x(1) - x(0)) / numberOfCategories[feature]; //@ts-ignore
    let rightFeature = undefined; //@ts-ignore
    let leftFeature = undefined;
    //same as for kde, only inversed 0, 1
    if (needGreyArea) {
        rightFeature = ancestorFeature.reduce((max: any, item: any) => {
            if (item.orientation === "right" && (!max || item.threshold > max.threshold)) {
                return item;
            }
            return max;
        }, null);
        rightFeature = (rightFeature == null ? 1 : scale(rightFeature.threshold))
        // get smallest ancestor with left as end point -> otherwise use 1
        leftFeature = ancestorFeature.reduce((min: any, item: any) => {
            if (item.orientation === "left" && (!min || item.threshold < min.threshold)) {
                return item;
            }
            return min;
        }, null);
        leftFeature = (leftFeature == null ? 0 : scale(leftFeature.threshold))
    }
    target.append("g")
        .attr("class", "feature-chart")
        .attr("transform", `translate( ${-size[0] / 2}, ${-size[1] / 2})`)
        .datum(histograms[feature])
        .selectAll("rect")
        .data((d: any) => d)
        .enter()
        .append("rect")
        .attr("fill", function (d: any) {
            //@ts-ignore
            if (needGreyArea && d.x0 <= rightFeature && d.x0 >= leftFeature) {
                return grayColor;
            }
            return color
        })
        .attr("opacity", ".8")
        .attr("stroke-linejoin", "round")
        .attr("x", function (d: any) {
            let xValue = x(d.x0) - (barWidth / 2);
            if (xValue < x(0)) {
                // prevent drawing below 0
                xValue = x(0)
            }
            return xValue
        })
        .attr("y", function (d: any) {
            return size[1] - yScale(d.length);
        })
        .attr("width", function (d: any) {
            let xValue = x(d.x0) - (barWidth / 2);
            if (xValue < x(0)) {
                // offset for first value 0
                return barWidth / 2
            }
            //prevent overdrawing right side
            barWidth = Math.min(nodeWidth - xValue, barWidth)
            return barWidth
        })
        .attr("height", function (d: any) {
            //@ts-ignore
            return yScale(d.length);
        })

    target.append("line")
        .attr("x1", function () {
            return x(scale(dataPoint.data.threshold))
        })
        .attr("x2", function () {
            return x(scale(dataPoint.data.threshold))
        })
        .attr("y1", 0)
        .attr("y2", size[1])
        .style("stroke-width", 2 * strokeScale)
        .style("stroke", "black")
        .style("fill", "none")
        .attr("transform", `translate( ${-size[0] / 2}, ${-size[1] / 2})`);
}


// after cluster slider change - give new clusters
export function showDecisionTreesForCluster(centroidTreeIndex: number, clusterNumber: number, cluster: any = null) {
    // we need to update the cluster after calling the slider
    if (cluster) {
        globalClusters = cluster;
    }

    d3.selectAll('.decisionTree').remove();
    const otherTrees = globalClusters["labels"].reduce((acc: any, val: number, index: number) => {
        //skip rep tree
        if (index === centroidTreeIndex) {
            return acc
        }
        if (val === clusterNumber) {
            acc.push(index);
        }
        return acc;
    }, []);

    //@ts-ignore
    createTree(globalTrees[centroidTreeIndex], centroidTreeIndex, true);

    for (let index of otherTrees) {
        //@ts-ignore
        createTree(globalTrees[index], index);
    }

    positionDecisionTrees();
}

export function positionDecisionTrees(): [number, number] {
    let offsetX = 20;
    let offsetY = 5;


    const mainSVG = d3.select("#decisionTrees").select("svg");
    const trees = mainSVG.selectAll("svg");

    //try and make a roughly rectangular area
    let totalArea = 0;
    trees.each(function (d, i) {
        const svg = d3.select(this);        //@ts-ignore
        const bbox = svg.node().getBBox();
        const width = bbox.width
        const height = bbox.height
        totalArea += width * height;
    });


    let totalWidth = Math.sqrt(totalArea);//getNodeLinkWindowWidth();


    offsetX = offsetX;

    let currentX = 0;
    let currentY = 0;
    let maxHeightInRow = 0;
    let maxWidth = 0;
    trees.each(function (d, i) {
        const svg = d3.select(this);
        //@ts-ignore
        const bbox = svg.node().getBBox();
        const width = bbox.width
        const height = bbox.height + bbox.y
        //check if we need to start a new row. Never start a row for the first line
        if ((currentX + width + offsetX * 2) > totalWidth) {
            maxWidth = Math.max(maxWidth, currentX + width);
            currentX = 0;
            currentY += maxHeightInRow + offsetY;
        }
        //transform to correct position
        svg.attr("x",currentX)
        svg.attr("y",currentY)
        svg.attr("width", width)
        svg.attr("height", height)
        currentX += width + offsetX;
        maxHeightInRow = Math.max(maxHeightInRow, height);

    })
    maxWidth = Math.max(maxWidth, currentX);

    let totalHeight = currentY + maxHeightInRow + offsetY;
    mainSVG.attr("width", maxWidth);
    mainSVG.attr("height", totalHeight);

    return [maxWidth, totalHeight]
}

// select a cluster from another view
export function selectClusterNodeLink(centroidTreeIndex: number) {
    for (let key of globalClusters["centroids"]) {
        if (key[1] === centroidTreeIndex) {
            let clusterNumber = key[0];

            showDecisionTreesForCluster(centroidTreeIndex, clusterNumber);
        }
    }
}


export function highlightNodeLinkFeature(featureIndex: number) {
    d3.selectAll(".nodeFeature-" + featureIndex).attr("stroke-width", 5 * strokeScale)
}

export function unHighlightNodeLinkFeature(featureIndex: number) {
    d3.selectAll(".nodeFeature-" + featureIndex).attr("stroke-width", 1 * strokeScale)
}

export function dimNodeLinkPaths(ranges: [number, number][], classes: boolean[][]) {
    //@ts-ignore
    d3.selectAll(".link").style("stroke", (d: d3.HierarchyLink<DecisionTreeNode>) => {
        if (typeof d.source.data.threshold === "string") {
            console.error("Threshold cannot be a string")
            return;
        }

        let feature = d.source.data.feature;
        let featureI = featureMap.get(feature)!;
        let splitPoint = d.source.data.threshold!;
        let range = ranges[featureI];
        let orientation = d.target.data.orientation;
        // if source is grey -> target and this should be grey  
        if (d.source.data.isGrey) {
            d.target.data.isGrey = true;
            return grayColor
        }
        // splitpoint is outside the range (left or right)
        if ((orientation === "left" && splitPoint < range[0]) || (orientation === "right" && splitPoint > range[1])) {
            d.target.data.isGrey = true;
            return grayColor
        }
        //The class is not selected
        if (!inSelectedClasses(d.target.data.allPredictions!, classes)) {
            d.target.data.isGrey = true;
            return grayColor
        }

        // splitpoint is inside        
        if (splitPoint > range[0] && splitPoint < range[1]) {
            d.target.data.isGrey = false;
            return blendColors(d.target.data.values)
        }

        d.target.data.isGrey = false; //we can arrive at the child.
        return blendColors(d.target.data.values)
    });
    //@ts-ignore
    d3.selectAll(".node").style("opacity", (d: d3.HierarchyNode<DecisionTreeNode>) => {
        if (d.data.isGrey) {
            return 0.35;
        } else {
            return 1;
        }
    });


}
/**
 * Returns whether predictions contains at least one of the specified classes
 * @param predictions 
 * @param classes 
 * @returns 
 */
function inSelectedClasses(predictions: Predictions, classes: boolean[][]): boolean {
    for (let classI = 0; classI < classes.length; classI++) {
        //No predictions from I
        if (predictions[classI] == undefined) {
            continue;
        }
        for (let classJ = 0; classJ < classes.length; classJ++) {
            //No predictions from I to J
            if (predictions[classJ] == undefined) {
                continue;
            }
            //class wasn't selected
            if (!classes[classI][classJ]) {
                continue;
            }
            //at least 1 prediction in one of the selected classes
            if (predictions[classI][classJ] > 0) {
                return true;
            }
        }
    }
    return false;
}

function addPredictionsToTreeData(treeData: DecisionTreeNode[], classes: Classes) {
    let classCount = Object.keys(classes).length;
    for (let treeRoot of treeData) {
        let allPredictions: Predictions = { count: 0 }
        for (let i = 0; i < classCount; i++) {
            allPredictions[i] = { count: 0 }
            for (let j = 0; j < classCount; j++) {
                allPredictions[i][j] = 0;
            }
        }

        //insert all correct predictions
        if (treeRoot.correctPredictionsClassesCounts != undefined) {
            for (let [featureName, count] of Object.entries(treeRoot.correctPredictionsClassesCounts)) {
                let featureIndex = classes[featureName].index;
                allPredictions[featureIndex][featureIndex] = count;
                allPredictions[featureIndex].count += count;
                allPredictions.count += count;
            }
        } else {//can be a leaf node, these don't have 
            if (treeRoot.feature == "Prediction") {
                let featureName = treeRoot.threshold;
                let featureIndex = classes[featureName].index;
                let count = treeRoot.correctPredictions;
                allPredictions[featureIndex][featureIndex] = count;
                allPredictions[featureIndex].count += count;
                allPredictions.count += count;
            }
        }
        //insert all wrong predictions
        for (let i = 0; i < classCount; i++) {
            if (treeRoot.wrongPredictions[i] == undefined) {
                continue;
            }
            for (let j = 0; j < classCount; j++) {
                if (treeRoot.wrongPredictions[i][j] == undefined) {
                    continue;
                }
                let count = treeRoot.wrongPredictions[i][j];
                allPredictions[i][j] = count;
                allPredictions[i].count += count;
                allPredictions.count += count;
            }
        }

        treeRoot.allPredictions = allPredictions;
        //recurse
        for (let child of treeRoot.children) {
            addPredictionsToTreeData([child], classes);
        }
    }

}