!#/bin/bash

#pushd ~/Projects/minecraft_AI/MalmoPlatform/MalmoEnv
pushd ~/Projects/minecraft_AI
python3 -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
popd
