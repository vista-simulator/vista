# read-dir-files [![Build Status](https://secure.travis-ci.org/mmalecki/node-read-dir-files.png)](http://travis-ci.org/mmalecki/node-read-dir-files)

Module for recursively (and non-recursively) reading all files in a directory.

## Installation

    npm install read-dir-files

## Usage
```js
  var readDirFiles = require('read-dir-files');
  readDirFiles('directory', function (err, files) {
    if (err) return console.dir(err);
    console.dir(files);
  });

```

### readDirFiles(dir, encoding, recursive, callback)

Parameters:

  * `dir` Directory to read files from
  * `encoding` Files encoding. **Optional.**
  * `recursive` Recurse into subdirectories? **Optional**, default `true`.
  * `callback` Callback. **Optional.**

Asynchronously reads all files from `dir` and returns them to the callback
in form:

```json
{
   dir: {
     file0: <Buffer ...>,
     file1: <Buffer ...>,
     sub: {
      file0: <Buffer ...>
    }
  }
}
```

If you pass it the encoding, instead of buffers you'll get strings.

