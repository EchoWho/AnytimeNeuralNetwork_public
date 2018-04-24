# Must make sure that you have azcopy, az-cli and philly-fs installed

# This needs to be run once every 14 days according to docs.
az login

# retrieve the SAS token 
mySAS=?$(az keyvault secret show --vault-name phillymsrlabs --name eu1 | jq .value | tr -d \" | cut -d "?" -f 2)

# test the SAS
wget https://phillyeustorage.blob.core.windows.net/msrlabs/hello.txt$mySAS -O ./hello.txt
if [[ $? -eq 0 ]]; then echo "success"; else echo "failed"; fi

# Copy code to azure blob
azcopy --source /home/dedey/AnytimeNeuralNetwork --destination https://phillyeustorage.blob.core.windows.net/msrlabs/dedey/AnytimeNeuralNetwork_master/ --recursive --dest-sas $mySAS --parallel-level 16

# Copy code from azure blob to gfs
export PHILLY_VC=msrlabs
# export AZURE_STORAGE_ACCESS_KEY=foobar # Only required if you use personal blob
/home/dedey/Dropbox/philly-fs/linux/philly-fs -cp -r https://phillyeustorage.blob.core.windows.net/msrlabs/dedey/AnytimeNeuralNetwork_master gfs://eu1/msrlabs/dedey/AnytimeNeuralNetwork_master
 