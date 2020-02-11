For npm scripts, NPM supports env variables in the form $npm_package_config_var, for example.  This can be used in `npm run` commands.  However, this is platform specific.  Windows uses %var% where \*nix uses $var.  This tool can be run like:

> compat "ncp $sourcePath $destPath"

so that, on windows platforms, the following command will be run:

> ncp %sourcePath% %destPath%

It will also allow relative paths to be used in the command on windows (arguments will be left alone as they generally work with either forward or back slashes):

> compat '../../node_modules/.bin/browserify index.js -s JID -o bundle.js'

=>

> ..\..\node_modules\.bin\browserify index.js -s JID -o bundle.js




