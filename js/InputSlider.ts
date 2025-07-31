import * as d3 from "d3";
import { asyncInit, prepareDepthAndBarPlot } from './index'
import { createProjection } from './Projection'
import { readCluster } from "./DataLoader";
import {showLoading, hideLoading} from "./index"
export function createInputSliders(data: any) {
    const sliderData = [];
    // for several area charts
    /*for (let key in data) {
        let tempArray = [];
        for (let step in data[key]) {
            tempArray.push({"value": data[key][step]["n_cluster"], "key": Number(step)})
        }
        let sliderEntry = {"key": key, "values": tempArray};
        sliderData.push(sliderEntry);
    }*/
    let tempArray = [];
    for (let step in data["cluster"]["minClusterSize"]) {
        tempArray.push({ "value": data["cluster"]["minClusterSize"][step]["n_cluster"], "key": Number(step) })
    }
    let sliderEntry = { "key": "minClusterSize", "values": tempArray };
    sliderData.push(sliderEntry);

    const stageHeight = 130;
    const stageWidth = 250;

    const spacing = 35; // Adjust the spacing between sliders
    const numberOfSliders = 1;
    const view = '#area-chart-container';
    d3.select(view).selectAll("*").remove();
    for (let i = 0; i < numberOfSliders; i++) {
        let yOffset = (spacing * i) + 35;
        areaChart(view, sliderData[i], data, stageWidth, stageHeight / numberOfSliders, yOffset);
    }
}

function areaChart(view: any, sliderData: any, responseData: any, width: number, height: number, yOffset: number) {
    const margin = { top: 15, right: 15, bottom: 35, left: 35 };
    const data = sliderData["values"];

    // append the svg object to the body of the page
    const svg = d3.select(view)
        .append("svg")
        .attr("class", "areaChart")
        .attr("width", width)
        .attr("height", height);

    const xmin = d3.min(data, function (d: any) { return d.key; });
    const xmax = d3.max(data, function (d: any) { return d.key; });
    const ymin = d3.min(data, function (d: any) { return d.value; });
    const ymax = d3.max(data, function (d: any) { return d.value; });

    // Add X axis
    const x = d3.scaleLinear()
        //@ts-ignore
        .domain([xmin, xmax])
        .range([margin.left, width - margin.right]);
    svg.append("g")
        .attr("transform", `translate(0,${height - margin.bottom + 5})`)
        .call(d3.axisBottom(x))
        .attr("font-size", "10px");

    // Add Y axis
    //@ts-ignore
    const y = d3.scaleLinear()
        //@ts-ignore
        .domain([ymin, ymax])
        .range([height - margin.bottom, margin.top]);
    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
         //@ts-ignore
        .call(d3.axisLeft(y).ticks(Math.floor((ymax - ymin) / 10)))
        //.call(d3.axisLeft(y))
        .attr("font-size", "10px");

    // Add the area
    svg.append("path")
        .datum(data)
        .attr("fill", "#d6dfe3")
        .attr("stroke", "#909fa7")
        .attr("stroke-width", 1.5)
        .attr("d", d3.area()
            .x(function (d: any) { return x(d.key) })
            .y0(y(0))
            .y1(function (d: any) { return y(d.value) })
        );


    const initValue = responseData.cluster["elbow"];
    const cx = x(initValue);
    const yValue = getYValueFromXPosition(initValue, x, data);
    const yPos = y(yValue);

    svg.append("text")
        .attr("id", "y-axis-label")
        .attr("transform", "rotate(-90)") // Rotate the text
        .attr("y", 0) // Position the text along y-axis
        .attr("x", 0 - (height / 2)) // Position the text along x-axis
        .attr("dy", "1em") // Adjust vertical position
        .style("text-anchor", "middle") // Center the text
        .attr("font-size", "12px")
        .text("#clusters");

    svg.append("text")
        .attr("id", "y-value-label")
        .attr("x", cx + 8)
        .attr("y", yPos - 17)
        .attr("dy", "1em") // Adjust vertical position
        .style("text-anchor", "middle") // Center the text
        .attr("font-size", "12px")
        .text(yValue);

    svg.append("text")
        .attr("id", "x-axis-label")
        .attr("text-anchor", "end")
        .attr("x", width)
        .attr("y", height)
        .attr("font-size", "12px")
        .text("Min Cluster Size: " + initValue);

    // Define variables for the draggable line
    let dragLine: any, dragHandle: any;
    // Create the draggable line
    dragLine = svg.append("line")
        .attr("class", "drag-line")
        .attr("x1", cx) // Initial position
        .attr("y1", height - 15)
        .attr("x2", cx)
        .attr("y2", margin.top)
        .attr("stroke", "grey")
        .attr("stroke-width", 2);

    dragHandle = svg.append("circle")
        .attr("class", "drag-handle")
        .attr("cx", cx) // Initial position
        .attr("cy", yPos)
        .attr("r", 6)
        .attr("fill", "white")
        .attr("stroke", "grey")
        .attr("stroke-width", 2)
        .style("cursor", "pointer");

    // Implement drag behavior for the line
    const dragHandler = d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);

    dragHandle.call(dragHandler);

    function dragStarted(event: any) {
        // Prevent propagation to avoid dragging the area chart underneath
        event.sourceEvent.stopPropagation();
    }

    function dragged(event: DragEvent) {
        // Update line position on drag
        const xPosition = Math.max(margin.left, Math.min(width - margin.right, event.x));
        let value = Math.round(x.invert(xPosition));

        let yValue = getYValueFromXPosition(value, x, data);
        let yPos = y(yValue);
        dragLine.attr("x1", xPosition).attr("x2", xPosition);
        dragHandle.attr("cx", xPosition).attr("cy", yPos);
        updateTextLabel(yValue, value, xPosition, yPos);
    }

    function dragEnded(event: DragEvent) {
        showLoading();
        // round xPosition to nearest value
        let xPosition = Math.max(margin.left, Math.min(width - margin.right, event.x));
        let value = Math.round(x.invert(xPosition));
        xPosition = x(value);
        let yValue = getYValueFromXPosition(value, x, data);
        let yPos = y(yValue);
        dragLine.attr("x1", xPosition).attr("x2", xPosition);
        dragHandle.attr("cx", xPosition).attr("cy", yPos);
        updateTextLabel(yValue, value, xPosition, yPos);
 
        // in order to show the loading animation
        setTimeout(async () => {
            await asyncInit(responseData,value);
            hideLoading();
        }, 0);

    }
}

function getYValueFromXPosition(xValue: number, x: any, data: any) {
    // Find corresponding y value
    const bisect = d3.bisector(function (d: any) { return d.key; }).right;
    let index = bisect(data, xValue);
    if (index === 0) {
        return data[0].value;
    }
    return data[index - 1].value;
}

function updateTextLabel(yValue: number, xValue: number,  xPosition: number, yPos: number) {
    d3.select("#x-axis-label").text("Min Cluster Size: " + xValue);
    
    d3.select("#y-value-label")
        .text(yValue)
        .attr("x", xPosition + 8) // Adjust as needed to position near the handle
        .attr("y", yPos - 17); // Adjust as needed for vertical positioning

}
