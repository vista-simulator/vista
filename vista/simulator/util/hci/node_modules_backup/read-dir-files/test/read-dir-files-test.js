var path = require('path'),
    assert = require('assert'),
    vows = require('vows'),
    readDirFiles = require('../');

vows.describe('read-dir-files').addBatch({
  'When using `read-dir-files`': {
    'and reading a directory (`readDirFiles("dir", cb)`)': {
      topic: function () {
        readDirFiles(path.join(__dirname, 'fixtures', 'dir'), this.callback);
      },
      'it should contain all files': function (data) {
        assert.isObject(data);
        assert.deepEqual(data, {
          a: new Buffer('Hello world\n'),
          b: new Buffer('Hello ncp\n'),
          c: new Buffer(''),
          d: new Buffer(''),
          sub: {
            a: new Buffer('Hello nodejitsu\n'),
            b: new Buffer('')
          }
        });
      }
    },
    'and reading a directory (`readDirFiles("dir", "utf8", cb)`)': {
      topic: function () {
        readDirFiles(
          path.join(__dirname, 'fixtures', 'dir'),
          'utf8',
          this.callback
        );
      },
      'it should contain all files': function (data) {
        assert.isObject(data);
        assert.deepEqual(data, {
          a: new Buffer('Hello world\n').toString('utf8'),
          b: new Buffer('Hello ncp\n').toString('utf8'),
          c: new Buffer('').toString('utf8'),
          d: new Buffer('').toString('utf8'),
          sub: {
            a: new Buffer('Hello nodejitsu\n').toString('utf8'),
            b: new Buffer('').toString('utf8')
          }
        });
      }
    },
    'and non-recursively reading a directory (`readDirFiles("dir", false, cb)`)': {
      topic: function () {
        readDirFiles(
          path.join(__dirname, 'fixtures', 'dir'),
          false,
          this.callback
        );
      },
      'it should contain all files': function (data) {
        assert.isObject(data);
        assert.deepEqual(data, {
          a: new Buffer('Hello world\n'),
          b: new Buffer('Hello ncp\n'),
          c: new Buffer(''),
          d: new Buffer('')
        });
      }
    }
  }
}).export(module);

