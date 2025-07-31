import * as druid from "@saehrimnir/druidjs";

//Javascript as druidjs does not have types

/**
 * Returns an ordering for the matrix that minimizes the distances of between the elements.
 * @param {} distanceMatrix Should be a number[][] containing the distances between the elements.
 */
export function get1dOrder(distanceMatrix) {
    if (distanceMatrix.length == 0) {
        console.error("Empty error given to get1dOrder.");
    }
    if (distanceMatrix.length == 1) {
        return [0];
    }


    let matrix = druid.Matrix.from(distanceMatrix);
    let method = "MDS"

    //console.log("Find the right dimensionreduction method, take care of speed")
    let positions = druid[method].transform(matrix, { d: 1 })._data//get one dimensional positions. Still needs to be turned into an order.
    let sortedPositions = [...positions].sort((a, b) => { return a - b }); //sort the array and make sure to not modify it.

    let order = [];
    for (let i = 0; i < positions.length; i++) {
        let index = sortedPositions.indexOf(positions[i]);
        //can have duplicates, indexof always finds the first, and they must be in order, so first open slot must be the right one.
        while (order[index] != undefined) {
            index = index + 1;
        }
        order[index] = i;
    }
    if (order.length != distanceMatrix.length) {
        console.error("1dOrder unsuccesfull. order length is " + order.length + " while there were " + distanceMatrix.length + " nodes to sort. Original distance matrix was: ")
        printMatrix(distanceMatrix);
    }


    for (let i = 0; i < order.length; i++) {
        if (order[i] == undefined) {
            console.error("undefined order in orderer")
            console.error(positions);
            console.error(sortedPositions);
            throw "1dimensional reduction is not working properly"
        }
    }

    return order
}


function printMatrix(matrix) {
    let string = "";
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix.length; j++) {
            let number = matrix[i][j];

            if (Number.isNaN(number)) {
                console.log("there is a NaN number in the matrix") //will be printed before the matrix
            }

            number = Math.round(number * 100) / 100; //only 2 digts
            let numberS = number.toString().padStart(5, " "); //ensure everything has the same width
            string += numberS + ","

        }
        string.slice(0, -1); //remove superflous ","
        string += "\n";
    }
}