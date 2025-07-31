import * as d3 from "d3";
import { focusFeature, initInteractionOriginalRanges, unfocusFeature, updateFeatureDistributionSelection } from "../Interactions";
import { kernelEpanechnikov, kernelDensityEstimator, silvermansRule } from "../Kernels";
import { trimTextWidth } from "../TextUtils";

let yAxes = {};
let categoricalRanges = {};
let originalValueScales = {};
let originalValueScalesCat = {};
let brushedRanges: [number, number][] = []
let brushedRangesOriginal: [number, number][] = []
let featureMap: Map<string, number> = new Map();
let format = d3.format(".3f");
let globalFeatures = {};
let brushes: d3.BrushBehavior<unknown>[] = [];
let brushAreas: d3.Selection<SVGGElement, unknown, null, undefined>[] = [];

const featureHeight = 30;
const axisHeight = 30
const rectangleWidth = 20
const nameWidth = 85;
const padding = 2;

let brushedInitialized: boolean; //as we are using call to set the initial brushed, we need to not trigger all the updates of elements not present yet.
export function createFeatureChart(target: string, features: any, colors: (t: number) => string) {
    brushedInitialized = false;
    initFeatureChart(features);



    d3.select(target).selectAll('*').remove();
    let values = Object.entries(features);
    let width = 230;


    let featureStartX = rectangleWidth + padding + nameWidth + padding;
    let featureWidth = width - featureStartX;


    //add offset
    let scrollOffset = 20;

    if (featureWidth <= 20) {
        console.error("Featurewidth of FeatureChart is too small to see the feature")
    }

    let height = (values.length * featureHeight) + axisHeight;
    let svg = d3.select(target)
        .append("svg")
        .attr("width", width + scrollOffset)
        .attr("height", height);

    let tooltip = d3.select(target)
        .append("div")
        .attr("id", "tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden");

    let g = svg.append("g");
    // add the x Axis
    let x = d3.scaleLinear()
        .domain([0, 1])
        .range([featureStartX, width]);



    // Plot the area
    let gs = g.selectAll(".feature-chart")
        .data(values)
        .enter()
        .append("g")
        .attr("transform", (d, i) => "translate(0, " + (i * featureHeight) + ")")
        .on("mouseenter", function (event: any, d: any) {
            focusFeature(d.index);
        })
        .on("mouseleave", function (event: any, d: any) {
            unfocusFeature(d.index);
            hideToolTip(tooltip)
        })
        .on("mousemove", function (event: any, d: any) {
            let xCoord = d3.pointer(event)[0];
            if (xCoord < featureStartX) {
                hideToolTip(tooltip);
            } else {
                showToolTip(tooltip, xCoord, x, event, d)
            }
        });

    //cover the unpainted areas
    gs.append("rect").attr("width", width)
        .attr("height", featureHeight)
        .attr("fill", "white")


    //add Feature text
    gs.append("text")
        .attr("class", "bar-text")
        .attr("x", rectangleWidth + padding)
        .attr("font-size", "15px")
        .attr("y", 20)
        .text(function (d: any) {
            return trimTextWidth(svg, d[0], "15px", nameWidth);
        })
        .append("title").text((d: any) => d[0]);

    //add KDE and Histograms
    gs.each(function (d: any, i: number) {
        this.classList.add("featureNumber" + i);//@ts-ignore
        d.index = i; //Add an index to the datum so we can grab it while hovering
        let self = d3.select(this);
        const index = d[0];
        let isCategorical = d[1].is_categorical;

        if (isCategorical) {
            drawHistogram(self, d, index, colors(i / gs.size()), x, featureHeight, width)
        } else {
            drawKDE(self, d, index, colors(i / gs.size()), x, featureHeight)
        }
    })

    brushes = []
    for (let i = 0; i < values.length; i++) {
        let brush = d3.brushX()
            .extent(function (d, i) {
                return [[x(0), 0], [x(1), featureHeight]]
            })
            .on("brush", brushed)
            .on("end", brushended);
        brushes[i] = brush;
    }

    brushAreas = []
    //add brush
    let brushesG = gs.append("g")
        .attr("class", "brush")
        .attr("id", (d, i) => "brush" + i)
        .each(function (d, i) {//@ts-ignore
            let brushArea = d3.select(this);
            brushAreas[i] = brushArea;
            brushArea.call(brushes[i])
            // cover the full area on init
            brushArea.call(brushes[i].move, [x(0), x(1)]);
        })


    //add the rectangle with the color.
    gs.append("rect")
        .attr("class", "bar")
        .attr("width", rectangleWidth)
        .attr("height", featureHeight)
        .attr("fill", (d, i) => colors(i / values.length))
        .style("cursor", "pointer")
        .on("click", (event, d) => {//@ts-ignore
            let index = d.index;
            let range: [number, number] = [featureStartX, featureStartX + featureWidth]; //Select everything by default
            //select the entire
            if (brushedRanges[index][0] == 0 && brushedRanges[index][1] == 1) { //everything was selected. Select nothing
                range = [featureStartX, featureStartX + 0.01];
            }

            brushAreas[index].call(brushes[index].move, range)
        })

    //add bottom axis
    g.append("g").attr("font-size", "8px").attr("transform", "translate(0, " + (height - axisHeight) + ")").call(d3.axisBottom(x).ticks(2));



    //@ts-ignore
    function brushed(event, d) {
        // show real values on brush
        const { selection } = event;
        //@ts-ignore
        let scale = originalValueScales[d[0]];
        let leftValue = format(scale(x.invert(selection[0])));
        let rightValue = format(scale(x.invert(selection[1])));
        let displayText = leftValue + "-" + rightValue;
        // categorical data
        if (d[1].is_categorical) {
            let values = getOriginalValueCategoricalBrush(selection, d[0], d[1])
            displayText = values.join(", ")
        }
        tooltip.html(`${displayText}`);
        //fast enough that we can just do it on brushing -> if tree update is to slow use false
        handleBrushend(selection, d, x, true)
    }
    //@ts-ignore
    function brushended(event, d) {
        const { selection } = event;
        handleBrushend(selection, d, x, true);
    }
    brushedInitialized = true;
}

//@ts-ignore
function handleBrushend(selection: [number, number], d, x, isEnd = false) {
    let key = d[0];
    let featureIndex = featureMap.get(key)!;
    if (selection) {
        //@ts-ignore
        let leftValue = x.invert(selection[0]);
        let rightValue = x.invert(selection[1]);
        let leftOriginal = getOriginalValue(selection[0], key, x);
        let rightOriginal = getOriginalValue(selection[1], key, x);
        if (d[1].is_categorical) {
            leftOriginal = getOriginalValueCatNorm(selection[0], key, x);
            rightOriginal = getOriginalValueCatNorm(selection[1], key, x);
        }
        brushedRangesOriginal[featureIndex] = [leftOriginal, rightOriginal];
        brushedRanges[featureIndex] = [leftValue, rightValue];
    } else {
        //restor initial full brush
        brushedRanges[featureIndex] = [0, 1];
        //@ts-ignore
        brushedRangesOriginal[featureIndex] = globalFeatures[key]["range"];
        //for categorical data
        if (d[1].is_categorical) {
            //@ts-ignore
            brushedRangesOriginal[featureIndex] = [globalFeatures[key]["categories_norminal"][0], globalFeatures[key]["categories_norminal"][globalFeatures[key]["categories_norminal"].length - 1]]
        }
        brushAreas[featureIndex].call(brushes[featureIndex].move, [x(0), x(1)]);
    }
    // give normalized feature range of all brushes
    // give non normalized feature range - important for node link splitpoints
    if (brushedInitialized) { //first brush is just to set the highlight
        updateFeatureDistributionSelection(brushedRanges, brushedRangesOriginal);
    }
}


export function scrollToFeature(featureIndex: number) {
    //TODO: Ideally this should be an animated scroll.    
    let scrollElement: Element = <Element>d3.select("#features-collapse").node()!

    let featureY = featureIndex * featureHeight;
    let currentY = scrollElement.scrollTop;
    let maxVisibleY = currentY + scrollElement.clientHeight
    //need to scroll up
    if (currentY > featureY) {
        scrollElement.scrollTop = featureY;
        return;
    }
    //need to scroll down.
    if (maxVisibleY <= featureY) {
        scrollElement.scrollTop = featureY + featureHeight - scrollElement.clientHeight;
        return;
    }
}

function initFeatureChart(features: any) {
    // init in case of new dataset
    yAxes = {};
    categoricalRanges = {};
    originalValueScales = {};
    brushedRanges = []
    brushedRangesOriginal = []
    globalFeatures = features;
    featureMap = new Map();

    for (let feature of Object.keys(features)) {
        featureMap.set(feature, features[feature]["index"]);
        brushedRanges.push([0, 1]);
        if (features[feature]["range"] === undefined) {
            //@ts-ignore
            let ordRange = [features[feature]["categories_norminal"][0], features[feature]["categories_norminal"][features[feature]["categories_norminal"].length - 1]]
            //@ts-ignore
            brushedRangesOriginal.push(ordRange);
            //@ts-ignore
            originalValueScalesCat[feature] = d3.scaleLinear().domain([0, 1]).range(ordRange)
        } else {
            brushedRangesOriginal.push(features[feature]["range"])
        }
    }
    initInteractionOriginalRanges(brushedRangesOriginal);
}

function getOriginalValue(xCoord: any, index: any, x: any) {
    let normalizedValue = x.invert(xCoord);
    //@ts-ignore
    let scale = originalValueScales[index];
    return scale(normalizedValue);
}

function getOriginalValueCatNorm(xCoord: any, index: any, x: any) {
    let normalizedValue = x.invert(xCoord);
    //@ts-ignore
    let scale = originalValueScalesCat[index];
    return scale(normalizedValue);
}

function getOriginalValueCategorical(xCoord: any, index: any, d: any) {
    //@ts-ignore
    const ranges = categoricalRanges[index];
    // find the correct categorie based on rect, or return an empty string if  no rect hovered
    const categoryRange = ranges.find((range: any) => xCoord >= range.start && xCoord <= range.end);
    return categoryRange ? d["categories"][ranges.indexOf(categoryRange)] : "";
}

function getOriginalValueCategoricalBrush(selection: [number, number], index: any, d: any) {
    //@ts-ignore
    const ranges = categoricalRanges[index];
    // Find all categories
    const matchingRanges = ranges.filter(
        (range: any) => ((selection[0] >= range.start && selection[0] <= range.end) ||
            (selection[1] >= range.start && selection[1] <= range.end) ||
            (selection[0] <= range.start && selection[1] >= range.end)));

    // Map the matching ranges to the corresponding category names
    const matchingCategories = matchingRanges.map((range: any) => d["categories"][ranges.indexOf(range)]);
    return matchingCategories;
}

function drawKDE(target: any, d: any, index: string, color: string, x: any, featureHeight: number) {
    target.append("path")
        .attr("class", "feature-chart")
        .attr("fill", color)
        .datum(function () {
            let bandwidth = silvermansRule(d[1]["scaled_values"], 1);
            let kde = kernelDensityEstimator(kernelEpanechnikov(bandwidth), x.ticks(100));
            let kdeValues = kde(d[1]["scaled_values"]);
            let yAxis = d3.scaleLinear()
                .range([featureHeight, 0])
                .domain([0, Math.max(...kdeValues.map((a: any) => a[1]))]);
            //@ts-ignore
            yAxes[index] = yAxis;
            let originalValueScale = d3.scaleLinear()
                .range(d[1]["range"])
                .domain([0, 1]);
            //@ts-ignore
            originalValueScales[index] = originalValueScale;

            return kdeValues;
        })
        .attr("opacity", ".8")
        .attr("stroke-linejoin", "round")
        .attr("d", d3.area()
            .curve(d3.curveBasis)
            .x(function (d: any) { return x(d[0]); })
            .y1(function (d: any) {
                //@ts-ignore
                let scale = yAxes[index];
                return scale(d[1]);
            })
            .y0(function () {
                //@ts-ignore
                let scale = yAxes[index];
                return scale(0);
            }))
}

function drawHistogram(target: any, d: any, index: string, color: string, x: any, featureHeight: number, width: number) {
    let numberOfCategories = d[1]["categories"].length;
    let barWidth = (x(1) - x(0)) / numberOfCategories;
    target.append("g")
        .attr("class", "feature-chart")
        .attr("fill", color)
        .datum(function () {
            let histogram = d3.histogram()
                .value((v: any) => v)
                .domain(x.domain())
                .thresholds(d[1]["numerical_categories"]);
            let bins = histogram(d[1]["scaled_values"]);
            let yAxis = d3.scaleLinear()
                .range([featureHeight, 0])
                .domain([0, Math.max(...bins.map((b: any) => b.length))]);
            //@ts-ignore
            yAxes[index] = yAxis;

            let originalValueScale = d3.scaleLinear()
                .range(d[1]["categories"])
                .domain(d[1]["numerical_categories"]);
            //@ts-ignore
            originalValueScales[index] = originalValueScale;
            return bins;
        })
        .selectAll("rect")
        .data((d: any) => d)
        .enter()
        .append("rect")
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
            //@ts-ignore
            let scale = yAxes[index];
            return scale(d.length);
        })
        .attr("width", function (d: any) {
            // save bar coords for mouseover
            //@ts-ignore
            if (!categoricalRanges[index]) {
                //@ts-ignore
                categoricalRanges[index] = []
            }
            let xValue = x(d.x0) - (barWidth / 2);
            if (xValue < x(0)) {
                // offset for first value 0
                //@ts-ignore
                categoricalRanges[index].push({ "start": x(0), "end": x(0) + (barWidth / 2) })
                return barWidth / 2
            }
            //prevent overdrawing right side
            barWidth = Math.min(width - xValue, barWidth)
            //@ts-ignore
            categoricalRanges[index].push({ "start": xValue, "end": xValue + barWidth })
            return barWidth
        })
        .attr("height", function (d: any) {
            //@ts-ignore
            let scale = yAxes[index];
            return featureHeight - scale(d.length);
        })
}


export function dimFeatureInFeatureDistributionPlot(featureNumber: number) {
    d3.selectAll(".featureNumber" + featureNumber).attr("opacity", 0.2)
}
export function undimFeatureInFeatureDistributionPlot(featureNumber: number) {
    d3.selectAll(".featureNumber" + featureNumber).attr("opacity", 1)
}

function hideToolTip(tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>) {
    tooltip.style("visibility", "hidden");
}


function showToolTip(tooltip: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>, xCoord: number, x: d3.ScaleLinear<number, number, never>, event: any, d: any) {
    let originalValue = format(getOriginalValue(xCoord, d[0], x))

    if (d[1].is_categorical) {
        originalValue = getOriginalValueCategorical(xCoord, d[0], d[1])
    }
    tooltip.style("visibility", "visible")
        .html(`${originalValue}`)
        .style("left", `${event.pageX + 10}px`)
        .style("top", `${event.target.getBoundingClientRect().y}px`);
}

