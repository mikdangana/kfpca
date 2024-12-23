#!/bin/bash

DIR=${0//provision_stack.sh/}
echo DIR = $DIR

# Source the OpenStack RC file for authentication
source $DIR/rrg-jacobsen-openrc.sh


create_instance() {
    # Define variables
    INSTANCE_NAME=$1 #"my-compute-instance"
    FLAVOR="c1-7.5gb-36"       # c1-7.5gb-36, c4-15gb-144 
    if [ $# -gt 1 ]; then FLAVOR=$2; fi
    IMAGE="Ubuntu-24.04-Noble-x64-2024-06"      # Name of the image to use (ensure it exists in your project)
    NETWORK="rrg-jacobsen-network" # Name of the network to attach
    KEYPAIR="devmd1"      # Name of the keypair for SSH access
    SECURITY_GROUP="michael-test"  # Name of the security group (default is commonly used)
    CINIT="cinit_worker.yaml"       # c1-7.5gb-36, c4-15gb-144 
    if [ $# -gt 2 ]; then CINIT=$3; fi

    if [ `openstack server show "$INSTANCE_NAME"` ]; then
        openstack server delete "$INSTANCE_NAME" --wait
    fi

    # Create the instance
    openstack server create \
      --flavor "$FLAVOR" \
      --image "$IMAGE" \
      --network "$NETWORK" \
      --security-group "$SECURITY_GROUP" \
      --key-name "$KEYPAIR" \
      --user-data "$CINIT" \
      "$INSTANCE_NAME"

}


create_stack() {
    #create_instance "ksurf-master" "c4-15gb-144" "cinit_master.yaml"
    MASTER_IP=`openstack server show ksurf-master | grep addresses | awk '{print $4}' | sed 's/.*=//'`
    echo MASTER_IP = $MASTER_IP
    sed -i 's/192.168.[0-9]*.[0-9]*/'${MASTER_IP}'/g' $DIR/cinit_producer_new.yaml
    sed -i 's/192.168.[0-9]*.[0-9]*/'${MASTER_IP}'/g' $DIR/cinit_worker.yaml
    #create_instance "ksurf-producer" "c4-15gb-144" "cinit_producer_new.yaml"
    for i in $(seq 2 1 2); do
        create_instance "ksurf-worker${i}" "c1-7.5gb-36" "cinit_worker.yaml"
    done
}


delete_stack() {
    openstack server delete "ksurf-master" --wait
    openstack server delete "ksurf-producer" --wait
    for i in $(seq 1 1 1); do
        openstack server delete "ksurf-worker${i}" --wait
    done
}


create_stack

# Check instance status
#openstack server list


