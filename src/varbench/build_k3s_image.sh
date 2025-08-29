export APPTAINER_CACHEDIR=/project/def-jacobsen/mdangana

apptainer cache clean --force 
apptainer build --fakeroot --ignore-fakeroot-command k3s_container.sif k3s.def
