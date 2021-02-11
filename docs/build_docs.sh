cd ../acd
pdoc --html . --output-dir ../docs --template-dir .
cp -rf ../docs/acd/* ../docs/
rm -rf ../docs/acd
cd ../docs
python3 style_docs.py