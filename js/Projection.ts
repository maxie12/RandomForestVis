import * as d3 from "d3";
import { Cluster } from "./types";
import { highlightCluster, selectCluster, unhighlightCluster } from "./Interactions";

interface projectionInfo {
  projection: [number, number]
  cluster_label: number;
  isCentroid: boolean
}

export function createProjection(projectionData: [[number, number]], cluster: Cluster) {
  let projection = addCentroidInformation(projectionData, cluster.centroids, cluster.labels);
  d3.select('#projection-container').selectAll('*').remove();
  
  const container = document.getElementById("projection-container");
  const containerBottom = container!.getBoundingClientRect().bottom; // Bottom position of container
  const viewportHeight = window.innerHeight; // Height of the viewport
  const spaceBelow = viewportHeight - containerBottom;

  let width = 250;
  let height = spaceBelow;
  let r = 5;

  let xValues = projection.map((d: any) => d["projection"][0]);
  let yValues = projection.map((d: any) => d["projection"][1]);
  const padding = 0.2;
  let xmin = Math.min(...xValues) - padding;
  let xmax = Math.max(...xValues) + padding;
  let ymin = Math.min(...yValues) - padding;
  let ymax = Math.max(...yValues) + padding;

  let xScale = d3.scaleLinear().range([r, width - r]).domain([xmin, xmax]);
  let yScale = d3.scaleLinear().range([r, height - r]).domain([ymin, ymax]);

  let svg = d3.select("#projection-container")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  let g = svg.append("g");

  for (let i = 1; i <= cluster["n_cluster"]; i++) {
    let points: [number, number][] = projection.filter((x: any) => x["cluster_label"] === i)
      .map((x: any) => [xScale(x["projection"][0]), yScale(x["projection"][1])]);

    let hull = d3.polygonHull(points)
    // check if less than 3 points
    if (hull !== null) {
      const hullg = g.append('g').style('opacity', 0.25)

      hullg.append("path")
        .attr("class", "hull-" + i)
        .style("stroke", "#000")
        .style("stroke-width", "26")
        .style('stroke-linecap', 'round')
        .style('stroke-linejoin', 'round')
        .style("fill", "#000")
        .attr("d", `M${hull.join("L")}Z`)

      hullg.append("path")
        .attr("class", "hull-" + i)
        .style("stroke", "#d6dfe3")
        .style("stroke-width", "25")
        .style('stroke-linecap', 'round')
        .style('stroke-linejoin', 'round')
        .style("fill", "#d6dfe3")
        .attr("d", `M${hull.join("L")}Z`)
        .on("mouseover", function () {
          g.selectAll(".cluster-" + i)
            .interrupt()
            .transition()
            .style("fill-opacity", function (x: any) {
              if (x["isCentroid"]) {
                return 1;
              }
              return 0.75;
            });
          highlightCluster(cluster.centroids.get(i)!);
        })
        .on("mouseout", function () {
          unhighlightCluster(cluster.centroids.get(i)!)
          g.selectAll("circle")
            .transition()
            .style("fill-opacity", function (x: any) {
              if (x["isCentroid"]) {
                return 1;
              }
              return 0.3;
            });
        })
        .on("click", function () {
          selectCluster(cluster.centroids.get(i)!);
        })
    }
  }

  g.selectAll('circle')
    .data(projection)
    .enter().append('circle')
    .attr("cx", (d: any) => xScale(d["projection"][0]))
    .attr("cy", (d: any) => yScale(d["projection"][1]))
    .attr('r', r)
    .attr('class', (d: any, i: number) => "cluster-" + d["cluster_label"] + " tree-" + i)
    .style("fill", '#94a2aa')
    .style("fill-opacity", function (x: any) {
      if (x["isCentroid"]) {
        return 1;
      }
      return 0.3;
    })
    .attr("stroke", "white")
    .style('pointer-events', 'none')
    .append("title")
    .text((d, i) => i);


  const zoom = d3.zoom().on("zoom", e => {
    let transform = e.transform
    g.attr("transform", transform);
  });

  //@ts-ignore
  svg.call(zoom).call(zoom.transform, d3.zoomIdentity);
}




function addCentroidInformation(projection: [[number, number]], centroids: Map<number, number>, labels: number[]): projectionInfo[] {
  let projection_with_cluster: projectionInfo[] = []
  for (let i = 0; i < projection.length; i++) {
    let isCentroid = false;
    if (centroids.get(labels[i]) == i) {
      isCentroid = true;
    }
    projection_with_cluster.push({ "projection": projection[i], "cluster_label": labels[i], "isCentroid": isCentroid });
  }
  return projection_with_cluster;
}

export function selectClusterProjection(treeIndex: number = -1) {
  //TODO show selected cluster again after mouseout?
  let tree = d3.selectAll('.tree-' + treeIndex).data();
  let clusterIndex = -1;
  if (tree.length > 0) {
    //@ts-ignore
    clusterIndex = tree[0]["cluster_label"];
  }

  d3.select("#projection-container").selectAll("circle")
    .transition()
    .style("fill-opacity", function (x: any) {
      if (x["isCentroid"]) {
        return 1;
      }
      return 0.3;
    });

  d3.selectAll(".cluster-" + clusterIndex)
    .interrupt()
    .transition()
    .style("fill-opacity", function (x: any) {
      if (x["isCentroid"]) {
        return 1;
      }
      return 1;
    });
}