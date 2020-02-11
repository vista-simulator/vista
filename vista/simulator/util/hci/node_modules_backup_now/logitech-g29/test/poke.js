//----------
// Includes
//----------
const color = require('../code/color.js')
const g     = require('../code/index.js')

//----------------
// Intended Usage
//----------------
console.log(color.cyan('Intended Usage'))
console.log()
console.log('    node')
console.log('    .load poke.js')
console.log('    bits(data[0]) ' + color.gray('<- where 0 (thru 11) is the buffer item to inspect'))
console.log()

//-----------
// Functions
//-----------
function bits(x) {
    // Based on https://github.com/sorensen/node-bitarray/blob/master/index.js -> BitArray.parse
    /**
    * Cast a 32bit integer or an array or buffer of 32bit integers into a bitmap
    * array, ensuring that they are a full octet length if specified
    *
    * @param {Number|Array|Buffer} 32bit integer or array or buffer of 32bit ints
    * @return {Array} bitmap array
    */

    let data = []
    let tmp = x

    if (typeof x === 'undefined') {
        return data
    }

    // Check for binary string
    if (typeof x === 'string') {
        for (let i = 0; i < x.length; i++) {
            data.push(+x[i])
        }
        return data.reverse()
    }

    // Check for single 32bit integer
    if (typeof x === 'number') {
        while (tmp > 0) {
            data.push(tmp % 2)
            tmp = Math.floor(tmp / 2)
        }
        //oct && (data = octet(data))
        data = octet(data)
        return data.reverse()
    }

    // Check for direct bit array
    if (Array.isArray(x)) {
        return x
    }

    // Assumed to be array / buffer of 32bit integers
    for (let i = 0; i < x.length; i++) {
        data = data.concat(bits(x[i]))
    }

    return data
} // bits

function octet(arr) {
    // Based on https://github.com/sorensen/node-bitarray/blob/master/index.js -> BitArray.octet
    /**
     * Ensure the given array is in the form of an octet, or, has
     * a length with a multiple of 8, zero fill missing indexes
     *
     * @param {Array} target
     * @return {Array} zero filled octet array
     */
    const len = arr.length
    const fill = len + (8 - len % 8)

    if (len !== 0 && len % 8 === 0) {
        return arr
    }

    for (let i = len; i < fill; i++) {
        arr[i] = 0
    }

    return arr
} // octet

//------------------
// Graceful Exiting
//------------------
process.on('SIGINT', function() {
    g.disconnect()
    process.exit()
})

//-----------
// Variables
//-----------
let data = ''

//---------
// Connect
//---------
g.connect(function(err) {
    g.on('data', function(val) {
        console.log(val)
        data = val
    })
})