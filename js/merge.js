const fs = require('fs').promises;
// var print = console.log;

// const svgPath = '3203.svg';

// async function test_composite() {
//   paper.setup(new paper.Size(1000, 1000));
//   const svgContent = (await fs.readFile(svgPath)).toString();

//   var g = paper.project.importSVG(svgContent);
//   g.removeChildren(0, 1);

//   var dPattern = /d="([^"]+)"/g;
//   var pathList = [];

//   for(var i = 0; i < 12; i ++) {
//     var pathString = g.children[i].exportSVG({asString: true});
//     var dString = [...pathString.matchAll(dPattern)][0][1];
//     var p = new paper.Path(dString);
//     pathList.push(p);
//   }

//   var primList = [];
//   var primFinal;
//   for(var i = 0; i < 6; i ++) {
//     var prim = pathList[i].subtract(pathList[i + 6]);
//     prim.fillColor = 'black';
//     if(primFinal === undefined) {
//       primFinal = prim;
//     } else {
//       primFinal = primFinal.unite(prim);
//     }
//   }
//   return primFinal.exportSVG({asString: true});
// }
var paper = require('paper');
const print = console.log;
paper.setup(new paper.Size(1000, 1000));

function get_dstring_from_path(svg) {
    // console.log('svg', svg)
    var dPattern = /d="([^"]+)"/g;
    var mid = [...svg.matchAll(dPattern)][0];
    if (mid == undefined) {
        return '';
    }
    // console.log(mid)
    var new_ds = mid[1];
    return new_ds;
}

function merge() {
    var d_strings = process.argv.slice(2);
    // console.log(d_strings.at(-1))

    var path_list = d_strings.map(ds => {
        var p = new paper.Path(ds);
        var p_string = p.exportSVG({asString: true});
        var new_ds = get_dstring_from_path(p_string);
        var p_new = new paper.Path(new_ds);
        return p_new;
    });

    var n_path = path_list.length / 2;
    var prim_final;
    for(var i = 0; i < n_path; i ++) {
        var prim = path_list[i].subtract(path_list[i + n_path]);
        prim.fillColor = 'black';
        if(prim_final === undefined) {
            prim_final = prim;
        } else {
            prim_final = prim_final.unite(prim);
        }
    }
    var res = get_dstring_from_path(prim_final.exportSVG({asString: true}));
    console.log(res);
}

merge();
