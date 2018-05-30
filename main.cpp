//
//  main.cpp
//  Flocking_MO
//
//  Created by adarsh kesireddy on 2/22/18.
//  Copyright © 2018 adarsh kesireddy. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>


using namespace std;

bool run_simulation = true;
bool test_simulation = true;

#define PI 3.14159265


/*************************
 Neural Network
 ************************/

struct connect{
    double weight;
};

static double random_global(double a) { return a* (rand() / double(RAND_MAX)); }

// This is for each Neuron
class Neuron;
typedef vector<Neuron> Layer;

class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    vector<connect> z_outputWeights;
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    unsigned z_myIndex;
    double z_outputVal;
    void setOutputVal(double val) { z_outputVal = val; }
    double getOutputVal(void) const { return z_outputVal; }
    void feedForward(const Layer prevLayer);
    double transferFunction(double x);
    
};

//This creates connection with neurons.
Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c) {
        z_outputWeights.push_back(connect());
        z_outputWeights.back().weight = randomWeight() - 0.5;
    }
    z_myIndex = myIndex;
}

double Neuron::transferFunction(double x){
    
    int case_to_use = 1;
    switch (case_to_use) {
        case 1:
            return tanh(x);
            break;
        case 2:
            //Dont use this case
            return 1/(1+exp(x));
            break;
        case 3:
            return x/(1+abs(x));
            break;
            
        default:
            break;
    }
    
    return tanh(x);
}

void Neuron::feedForward(const Layer prevLayer){
    double sum = 0.0;
    bool debug_sum_flag = false;
    
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        if(debug_sum_flag == true){
            cout<<prevLayer[n].getOutputVal()<<endl;
            cout<<&prevLayer[n].z_outputWeights[z_myIndex];
            cout<<prevLayer[n].z_outputWeights[z_myIndex].weight;
        }
        sum += prevLayer[n].getOutputVal() * prevLayer[n].z_outputWeights[z_myIndex].weight;
        //cout<<"This is sum value"<<sum<<endl;
    }
    z_outputVal = Neuron::transferFunction(sum);
}

//This is single neural network
class Net{
public:
    Net(vector<unsigned> topology);
    void feedForward(vector<double> inputVals);
    vector<Layer> z_layer;
    vector<double> outputvaluesNN;
    double backProp();
    double z_error;
    double z_error_temp;
    vector<double> z_error_vector;
    void mutate();
    vector<double> temp_inputs;
    vector<double> temp_targets;
    
    //CCEA
    double fitness;
    vector<double> closest_dist_to_poi;
    vector<double> global_objective_values;
    vector<double> local_objective_values;
    vector<double> difference_objective_values;
    
    //For team
    
    int my_team_number;
    
    //For team
    double local_reward_wrt_team;
    double global_reward_wrt_team;
    double difference_reward_wrt_team;
    
    //Store x and y values
    vector<double> store_x_values;
    vector<double> store_y_values;
    
    //Objective values
    double collision_with_agents;
    double collision_with_obstacles;
    double path_value;
    
    //NSGA-II
    vector<int> dominating_over;
    double dominating_me;
    bool already_has_front;
    int front_number;
    vector<double> crowding_distance;
    double crowding_distance_rank;
    vector<vector<double>> front;
    vector<vector<double>> dominate;
    vector<double> num_donimated;
    double rank;
    bool safe;
    
    
    //New sorting
    bool ranked;
    int best_rank;
    vector<double> threeD;
    
    //summation
    double summation;
};

Net::Net(vector<unsigned> topology){
    
    for(int  numLayers = 0; numLayers<topology.size(); numLayers++){
        //unsigned numOutputs = numLayers == topology.size() - 1 ? 0 : topology[numLayers + 1];
        
        unsigned numOutputs;
        if (numLayers == topology.size()-1) {
            numOutputs=0;
        }else{
            numOutputs= topology[numLayers+1];
        }
        
        if(numOutputs>30){
            cout<<"Stop it number outputs coming out"<<numOutputs<<endl;
            exit(10);
        }
        
        z_layer.push_back(Layer());
        
        for(int numNeurons = 0; numNeurons <= topology[numLayers]; numNeurons++){
            //cout<<"This is neuron number:"<<numNeurons<<endl;
            z_layer.back().push_back(Neuron(numOutputs, numNeurons));
        }
    }
}

void Net::mutate(){
    /*
     //popVector[temp].z_layer[temp][temp].z_outputWeights[temp].weight
     */
    for (int l =0 ; l < z_layer.size(); l++) {
        for (int n =0 ; n< z_layer.at(l).size(); n++) {
            for (int z=0 ; z< z_layer.at(l).at(n).z_outputWeights.size(); z++) {
                z_layer.at(l).at(n).z_outputWeights.at(z).weight += random_global(.5)-random_global(.5);
            }
        }
    }
}

void Net::feedForward(vector<double> inputVals){
    
    assert(inputVals.size() == z_layer[0].size()-1);
    for (unsigned i=0; i<inputVals.size(); ++i) {
        z_layer[0][i].setOutputVal(inputVals[i]);
    }
    for (unsigned layerNum = 1; layerNum < z_layer.size(); ++layerNum) {
        Layer &prevLayer = z_layer[layerNum - 1];
        for (unsigned n = 0; n < z_layer[layerNum].size() - 1; ++n) {
            z_layer[layerNum][n].feedForward(prevLayer);
        }
    }
    temp_inputs.clear();
    
    
    Layer &outputLayer = z_layer.back();
    z_error_temp = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        //cout<<"This is value from outputlayer.getourputvalue:::::"<<outputLayer[n].getOutputVal()<<endl;
        //double delta = temp_targets[n] - outputLayer[n].getOutputVal();
        //cout<<"This is delta value::"<<delta;
        //z_error_temp += delta * delta;
        outputvaluesNN.push_back(outputLayer[n].getOutputVal());
    }
    
}

double Net::backProp(){
    z_error = 0.0;
    for (int temp = 0; temp< z_error_vector.size(); temp++) {
        //cout<<"This is z_error_vector"<<temp<<" value::"<< z_error_vector[temp]<<endl;
        z_error += z_error_vector[temp];
    }
    //    cout<<"This is z_error::"<<z_error<<endl;
    return z_error;
}


double cal_distance(double x1, double y1, double x2, double y2){
    return sqrt(pow((x1 - x2), 2)+pow((y1 - y2), 2));
}




/***********************
 POI
 **********************/
class POI{
public:
    double x_position_poi,y_position_poi,value_poi;
    //Environment test;
    //vector<Rover> individualRover;
    vector<double> x_position_poi_vec;
    vector<double> y_position_poi_vec;
    vector<double> value_poi_vec;
};

/************************
 Environment
 ***********************/

class Environment{
public:
    vector<POI> individualPOI;
    vector<POI> group_1;
    vector<POI> group_2;
};

/************************
 Rover
 ***********************/

double resolve(double angle);


class Rover{
    //Environment environment_object;
public:
    double x_position,y_position;
    vector<double> x_position_vec,y_position_vec;
    vector<double> sensors;
    vector<Net> singleneuralNetwork;
    void sense_poi(double x, double y, double val);
    void sense_rover(double x, double y);
    double sense_poi_delta(double x_position_poi,double y_position_poi);
    double sense_rover_delta(double x_position_otherrover, double y_position_otherrover);
    double sense_obstacle(double obstacle_x_location, double obstacle_y_location, double rover_x, double rover_y);
    
    
    vector<double> controls;
    double delta_x,delta_y;
    
    double theta;
    double phi;
    double gamma;
    void reset_sensors();
    int find_quad(double x, double y);
    double find_phi(double x, double y);
    double find_theta(double x_sensed, double y_sensed);
    double find_gamma(double x_sensed, double y_sensed);
    void move_rover(double dx, double dy);
    
    double reward =0.0;
    void sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover);
    void sense_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover,vector<int>* p_index_number,int current_number,vector<Rover>* teamRover);
    double sense_rover_new(double current_x, double current_y, double x_position_otherrover, double y_position_otherrover);
    double sense_poi_new(double xposition, double yposition,double x_position_poi,double y_position_poi );
    int find_quad_new(double x,double y, double the, double x_sense, double y_sense);
    
    //stored values
    vector<double> max_reward;
    vector<double> policy;
    //vector<double> best_closest_distance;
    
    //Neural network
    vector<Net> network_for_agent;
    void create_neural_network_population(int numNN,vector<unsigned> topology);
    
    //random numbers for neural networks
    vector<int> random_numbers;
    
    void sense_values_all(int current_number,vector<Rover>* teamRover,vector<vector<double>>* ob_location);
    
    //destination location
    double destination_x_position;
    double destination_y_position;
    
    //New domain functions
    
    
};

// variables used: indiNet -- object to Net
void Rover::create_neural_network_population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        network_for_agent.push_back(singleNetwork);
    }
    
}

//Function returns: sum of values of POIs divided by their distance
double Rover::sense_poi_delta(double x_position_poi,double y_position_poi ){
    double delta_sense_poi=0;
    double distance = sqrt(pow(x_position-x_position_poi, 2)+pow(y_position-y_position_poi, 2));
    double minimum_observation_distance =0.0;
    delta_sense_poi=(distance>minimum_observation_distance)?distance:minimum_observation_distance ;
    return delta_sense_poi;
}

double Rover::sense_poi_new(double xposition, double yposition,double x_position_poi,double y_position_poi ){
    double delta_sense_poi=0;
    double distance = sqrt(pow(xposition-x_position_poi, 2)+pow(yposition-y_position_poi, 2));
    double minimum_observation_distance =0.0;
    delta_sense_poi=(distance>minimum_observation_distance)?distance:minimum_observation_distance ;
    return delta_sense_poi;
}

//Function returns: sum of sqaure distance from a rover to all the other rovers in the quadrant
double Rover::sense_rover_delta(double x_position_otherrover, double y_position_otherrover){
    double delta_sense_rover=0.0;
    if (x_position_otherrover == NULL || y_position_otherrover == NULL) {
        return delta_sense_rover;
    }
    double distance = sqrt(pow(x_position-x_position_otherrover, 2)+pow(y_position-y_position_otherrover, 2));
    delta_sense_rover=(1/distance);
    
    return delta_sense_rover;
}

double Rover::sense_rover_new(double current_x, double current_y, double x_position_otherrover, double y_position_otherrover){
    double delta_sense_rover=0.0;
    double distance = sqrt(pow(current_x-x_position_otherrover, 2)+pow(current_y-y_position_otherrover, 2));
    delta_sense_rover=(1/distance);
    
    return delta_sense_rover;
}


//Function returns: sum of distance from obstacle

void Rover::sense_poi(double poix, double poiy, double val){
    double delta = sense_poi_delta(poix, poiy);
    int quad = find_quad(poix,poiy);
    sensors.at(quad) += val/delta;
}

void Rover::sense_rover(double otherx, double othery){
    double delta = sense_rover_delta(otherx,othery);
    int quad = find_quad(otherx,othery);
    sensors.at(quad+4) += 1/delta;
}

double Rover::sense_obstacle(double obstacle_x_location, double obstacle_y_location, double rover_x, double rover_y){
    return sqrt(pow(obstacle_x_location - rover_x, 2) + pow(obstacle_y_location - rover_y,2));
}


void Rover::reset_sensors(){
    sensors.clear();
    for(int i=0; i<12; i++){
        sensors.push_back(0.0);
    }
}

double Rover::find_phi(double x_sensed, double y_sensed){
    double distance_in_x_phi =  x_sensed - x_position;
    double distance_in_y_phi =  y_sensed - y_position;
    double deg2rad = 180/PI;
    double phi = (atan2(distance_in_y_phi,distance_in_x_phi) *(deg2rad));
    
    return phi;
}

double Rover::find_theta(double x_sensed, double y_sensed){
    double distance_in_x_theta =  x_sensed - x_position;
    double distance_in_y_theta =  y_sensed - y_position;
    theta = atan2(distance_in_y_theta,distance_in_x_theta) * (180 / PI);
    
    return theta;
}

double Rover::find_gamma(double x_sensed, double y_sensed){
    double distance_in_x_gamma =  x_sensed - x_position;
    double distance_in_y_gamma =  y_sensed - y_position;
    gamma = (atan2(distance_in_y_gamma,distance_in_x_gamma) * (180 / PI));
    
    return gamma;
}

int Rover::find_quad_new(double x,double y, double the, double x_sense, double y_sense){
    int quadrant;
    
    double distance_in_x_phi =  x_sense - x;
    double distance_in_y_phi =  y_sense - y;
    double deg2rad = 180/PI;
    double phi = (atan2(distance_in_x_phi,distance_in_y_phi) *(deg2rad));
    double quadrant_angle = phi - the;
    
    quadrant_angle = resolve(quadrant_angle);
    assert(quadrant_angle != NAN);
    //    cout << "IN QUAD: FIND PHI: " << phi << endl;
    
    phi = resolve(phi);
    
    //    cout << "IN QUAD: FIND PHI2: " << phi << endl;
    
    int case_number;
    if ((0 <= quadrant_angle && 45 >= quadrant_angle)||(315 < quadrant_angle && 360 >= quadrant_angle)) {
        //do something in Q1
        case_number = 0;
    }else if ((45 < quadrant_angle && 135 >= quadrant_angle)) {
        // do something in Q2
        case_number = 1;
    }else if((135 < quadrant_angle && 225 >= quadrant_angle)){
        //do something in Q3
        case_number = 2;
    }else if((225 < quadrant_angle && 315 >= quadrant_angle)){
        //do something in Q4
        case_number = 3;
    }
    quadrant = case_number;
    
    //    cout << "QUADANGLE =  " << quadrant_angle << endl;
    //    cout << "QUADRANT = " << quadrant << endl;
    
    return quadrant;
}

int Rover::find_quad(double x_sensed, double y_sensed){
    int quadrant;
    
    double phi = find_phi(x_sensed, y_sensed);
    double quadrant_angle = phi - theta;
    quadrant_angle = resolve(quadrant_angle);
    assert(quadrant_angle != NAN);
    //    cout << "IN QUAD: FIND PHI: " << phi << endl;
    
    phi = resolve(phi);
    
    //    cout << "IN QUAD: FIND PHI2: " << phi << endl;
    
    int case_number;
    if ((0 <= quadrant_angle && 45 >= quadrant_angle)||(315 < quadrant_angle && 360 >= quadrant_angle)) {
        //do something in Q1
        case_number = 0;
    }else if ((45 < quadrant_angle && 135 >= quadrant_angle)) {
        // do something in Q2
        case_number = 1;
    }else if((135 < quadrant_angle && 225 >= quadrant_angle)){
        //do something in Q3
        case_number = 2;
    }else if((225 < quadrant_angle && 315 >= quadrant_angle)){
        //do something in Q4
        case_number = 3;
    }
    quadrant = case_number;
    
    //    cout << "QUADANGLE =  " << quadrant_angle << endl;
    //    cout << "QUADRANT = " << quadrant << endl;
    
    return quadrant;
}

void Rover::move_rover(double dx, double dy){
    
//    double aom = (atan2(dy,dx)*180/PI); /// angle of movement
//    double rad2deg = (PI/180);
//    x_position = x_position + sin(theta*rad2deg) * dy + cos(theta*rad2deg) * dx;
//    y_position = y_position + sin(theta*rad2deg) * dx + cos(theta*rad2deg) * dy;
//    theta = theta + aom;
//    theta = resolve(theta);
    
    //x_position =(x_position)+  (dy* cos(theta*(PI/180)))-(dx *sin(theta*(PI/180)));
    //y_position =(y_position)+ (dy* sin(theta*(PI/180)))+(dx *cos(theta*(PI/180)));
    //theta = theta+ (atan2(dx,dy) * (180 / PI));
    //theta = resolve(theta);
    
    x_position += dx;
    y_position += dy;
}


//Takes all poi values and update sensor values
void Rover::sense_all_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover){
    reset_sensors();
    
    double temp_delta_value = 0.0;
    vector<double> temp_delta_vec;
    int temp_quad_value =0;
    vector<double> temp_quad_vec;
    
    assert(x_position_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    assert(value_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    
    for (int value_calculating_delta = 0 ; value_calculating_delta < x_position_poi_vec_rover.size(); value_calculating_delta++) {
        temp_delta_value = sense_poi_delta(x_position_poi_vec_rover.at(value_calculating_delta), y_position_poi_vec_rover.at(value_calculating_delta));
        temp_delta_vec.push_back(temp_delta_value);
    }
    
    for (int value_calculating_quad = 0 ; value_calculating_quad < x_position_poi_vec_rover.size(); value_calculating_quad++) {
        temp_quad_value = find_quad(x_position_poi_vec_rover.at(value_calculating_quad), y_position_poi_vec_rover.at(value_calculating_quad));
        temp_quad_vec.push_back(temp_quad_value);
    }
    
    assert(temp_delta_vec.size()== temp_quad_vec.size());
    
    for (int update_sensor = 0 ; update_sensor<temp_quad_vec.size(); update_sensor++) {
        sensors.at(temp_quad_vec.at(update_sensor)) += value_poi_vec_rover.at(update_sensor)/temp_delta_vec.at(update_sensor);
    }
    
}


void Rover::sense_values(vector<double> x_position_poi_vec_rover,vector<double> y_position_poi_vec_rover,vector<double> value_poi_vec_rover,vector<int>* p_index_number,int current_number,vector<Rover>* teamRover){
    
    //First sense POIs
    double temp_delta_value = 0.0;
    vector<double> temp_delta_vec;
    int temp_quad_value =0;
    vector<double> temp_quad_vec;
    
    assert(x_position_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    assert(value_poi_vec_rover.size() == y_position_poi_vec_rover.size());
    
    for (int value_calculating_delta = 0 ; value_calculating_delta < x_position_poi_vec_rover.size(); value_calculating_delta++) {
        temp_delta_value = sense_rover_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position,x_position_poi_vec_rover.at(value_calculating_delta), y_position_poi_vec_rover.at(value_calculating_delta));
        temp_delta_vec.push_back(temp_delta_value);
    }
    
    for (int value_calculating_quad = 0 ; value_calculating_quad < x_position_poi_vec_rover.size(); value_calculating_quad++) {
        temp_quad_value = find_quad_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(current_number).theta, x_position_poi_vec_rover.at(value_calculating_quad), y_position_poi_vec_rover.at(value_calculating_quad));
        temp_quad_vec.push_back(temp_quad_value);
    }
    
    assert(temp_delta_vec.size()== temp_quad_vec.size());
    
    //Now sense rovers
    double new_temp_delta_value = 0.0;
    vector<double> new_temp_delta_vec;
    int new_temp_quad_value =0;
    vector<double> new_temp_quad_vec;
    
    for (int other_rover = 0; other_rover < p_index_number->size(); other_rover++) {
        if (current_number != other_rover) {
            //x and y coordinates
            new_temp_delta_value = sense_rover_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(other_rover).x_position, teamRover->at(other_rover).y_position);
            new_temp_delta_vec.push_back(new_temp_delta_value);
            new_temp_quad_value = find_quad_new(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, teamRover->at(current_number).theta,teamRover->at(other_rover).x_position, teamRover->at(other_rover).y_position);
            new_temp_quad_vec.push_back(new_temp_quad_value);
        }
    }
    
    assert(new_temp_delta_vec.size()== new_temp_quad_vec.size());
    
    for (int update_sensor = 0 ; update_sensor<temp_quad_vec.size(); update_sensor++) {
        sensors.at(temp_quad_vec.at(update_sensor)) += value_poi_vec_rover.at(update_sensor)/temp_delta_vec.at(update_sensor);
    }
    
    for (int update_sensor = 0 ; update_sensor<new_temp_quad_vec.size(); update_sensor++) {
        sensors.at(new_temp_quad_vec.at(update_sensor)) += new_temp_delta_vec.at(update_sensor);
    }
}

int quad_value(double x1, double y1, double x2, double y2){
    int quad_number = 999;
    
    double value_x = x1 - x2;
    double value_y = y1 - y2;
    double angle = (atan2(value_y, value_x)*(180/PI));
    
    angle = resolve(angle);
    
    if ((0 <= angle && 45 >= angle)||(315 < angle && 360 >= angle)) {
        //do something in Q1
        quad_number = 0;
    }else if ((45 < angle && 135 >= angle)) {
        // do something in Q2
        quad_number = 1;
    }else if((135 < angle && 225 >= angle)){
        //do something in Q3
        quad_number = 2;
    }else if((225 < angle && 315 >= angle)){
        //do something in Q4
        quad_number = 3;
    }
    
    assert(quad_number != 999);
    
    return quad_number;
}

void Rover::sense_values_all(int current_number,vector<Rover>* teamRover,vector<vector<double>>* ob_location){
    
    bool verbose = false;
    
    if (verbose) {
        for (int index = 0 ; index < teamRover->at(current_number).sensors.size(); index++) {
            cout<<teamRover->at(current_number).sensors.at(index)<<"\t";
        }
        cout<<endl;
    }
    
    
    //Other rover information
    for (int rover_number = 0 ; rover_number <teamRover->size(); rover_number++) {
        if (rover_number != current_number) {
            //First find distance to other rover
            if(verbose){
                cout<< "Rover"<<teamRover->at(rover_number).x_position<<"\t"<<teamRover->at(rover_number).y_position<<endl;
                cout<< "Current Rover"<<teamRover->at(current_number).x_position<<"\t"<<teamRover->at(current_number).y_position<<endl;
            }
            double distance = cal_distance(teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position, teamRover->at(current_number).x_position, teamRover->at(current_number).y_position);
            int quad = quad_value(teamRover->at(rover_number).x_position, teamRover->at(rover_number).y_position, teamRover->at(current_number).x_position, teamRover->at(current_number).y_position);
            teamRover->at(current_number).sensors.at(quad+4) += distance;
            
        }
    }
    
    if (verbose) {
        for (int index = 0 ; index < teamRover->at(current_number).sensors.size(); index++) {
            cout<<teamRover->at(current_number).sensors.at(index)<<"\t";
        }
        cout<<endl;
    }
    
    
    //destination values
    double destination_distance = cal_distance(teamRover->at(current_number).x_position,teamRover->at(current_number).y_position,teamRover->at(current_number).destination_x_position,teamRover->at(current_number).destination_y_position);
    
    int destination_quad = quad_value(teamRover->at(current_number).x_position,teamRover->at(current_number).y_position,teamRover->at(current_number).destination_x_position,teamRover->at(current_number).destination_y_position);
    teamRover->at(current_number).sensors.at(destination_quad) += destination_distance;
    
    if (verbose) {
        for (int index = 0 ; index < teamRover->at(current_number).sensors.size(); index++) {
            cout<<teamRover->at(current_number).sensors.at(index)<<"\t";
        }
        cout<<endl;
    }
    
    //obstacle values
    for (int index = 0 ; index < ob_location->size() ; index++) {
        double distance = cal_distance(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, ob_location->at(index).at(0), ob_location->at(index).at(1));
        int quad = quad_value(teamRover->at(current_number).x_position, teamRover->at(current_number).y_position, ob_location->at(index).at(0), ob_location->at(index).at(1));
        teamRover->at(current_number).sensors.at(quad+8) += distance;
    }
    
    if (verbose) {
        for (int index = 0 ; index < teamRover->at(current_number).sensors.size(); index++) {
            cout<<teamRover->at(current_number).sensors.at(index)<<"\t";
        }
        cout<<endl;
    }
    
    
}


/*************************
 Population
 ************************/
//This is for population of neural network
class Population{
public:
    void create_Population(int numNN,vector<unsigned> topology);
    vector<Net> popVector;
    void runNetwork(vector<double> inputVal,int number_neural);
    void sortError();
    void mutation(int numNN);
    void newerrorvector();
    void findindex();
    int returnIndex(int numNN);
    void repop(int numNN);
    
};

// variables used: indiNet -- object to Net
void Population::create_Population(int numNN,vector<unsigned> topology){
    
    for (int populationNum = 0 ; populationNum<numNN; populationNum++) {
        //cout<<"This is neural network:"<<populationNum<<endl;
        Net singleNetwork(topology);
        popVector.push_back(singleNetwork);
    }
    
}

//Return index of higher
int Population::returnIndex(int numNN){
    int temp = numNN;
    int number_1 = (rand() % temp);
    int number_2 = (rand() % temp);
    while (number_1 == number_2) {
        number_2 = (rand() % temp);
    }
    
    if (popVector[number_1].z_error<popVector[number_2].z_error) {
        return number_2;
    }else if (popVector[number_1].z_error>popVector[number_2].z_error){
        return number_1;
    }else{
        return NULL;
    }
}

void Population::repop(int numNN){
    for (int temp =0 ; temp<numNN/2; temp++) {
        int R = rand()% popVector.size();
        popVector.push_back(popVector.at(R));
        popVector.back().mutate();
    }
}

void Population::runNetwork(vector<double> inputVals,int num_neural){
    popVector.at(num_neural).feedForward(inputVals);
    popVector.at(num_neural).backProp();
}

/**************************
 Simulation Functions
 **************************/
// Will resolve angle between 0 to 360
double resolve(double angle){
    while(angle >= 360){
        angle -=360;
    }
    while(angle < 0){
        angle += 360;
    }
    while (angle == 360) {
        angle = 0;
    }
    return angle;
}


double find_scaling_number(vector<Rover>* teamRover, POI* individualPOI){
    double number =0.0;
    double temp_number =0.0;
    vector < vector <double> > group_sensors;
    
    for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
        for (int policy_number = 0; policy_number< individualPOI->value_poi_vec.size(); policy_number++) {
            teamRover->at(rover_number).reset_sensors();
            teamRover->at(rover_number).sense_poi(individualPOI->x_position_poi_vec.at(policy_number), individualPOI->y_position_poi_vec.at(policy_number), individualPOI->value_poi_vec.at(policy_number));
            group_sensors.push_back(teamRover->at(rover_number).sensors);
        }
    }
    
    assert(!group_sensors.empty());
    
    for (int i=0; i<group_sensors.size(); i++) {
        temp_number=*max_element(group_sensors.at(i).begin(), group_sensors.at(i).end());
        if (temp_number>number) {
            number=temp_number;
        }
    }
    
    
    assert(number != 0.0);
    
    return number;
}

/*****************************************************************
 Test Rover in environment
 ***************************************************************/

// Tests Stationary POI and Stationary Rover in all directions
bool POI_sensor_test(){
    bool VERBOSE = false;
    
    bool passfail = false;
    
    bool pass1 = false;
    bool pass2 = false;
    bool pass3 = false;
    bool pass4 = false;
    
    POI P;
    Rover R;
    
    /// Stationary Rover
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 0; /// north
    
    P.value_poi = 10;
    
    /// POI directly north, sensor 0 should read; no others.
    P.x_position_poi = 0.001;
    P.y_position_poi = 1;
    cout<<"values"<<endl;
    cout<<R.phi<<endl;
    cout<<R.theta<<endl;
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    cout<<"values"<<endl;
    cout<<R.phi<<endl;
    cout<<R.theta<<endl;
    for(int s=0;s<R.sensors.size();s++){
        cout<<R.sensors.at(s)<<"\t";
    }
    cout<<endl;
    
    if(R.sensors.at(0) != 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass1 = true;
    }
    
    assert(pass1 == true);
    
    if(VERBOSE){
        cout << "Direct north case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly south, sensor 2 should read; no others.
    P.x_position_poi = 0;
    P.y_position_poi = -1;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) !=0 && R.sensors.at(3) == 0){
        pass2 = true;
    }
    
    assert(pass2 == true);
    
    if(VERBOSE){
        cout << "Direct south case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    /// POI directly east, sensor 1 should read; no others.
    P.x_position_poi = 1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) != 0 && R.sensors.at(2) ==0 && R.sensors.at(3) == 0){
        pass3 = true;
    }
    
    assert(pass3 == true);
    
    if(VERBOSE){
        cout << "Direct east case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    
    
    /// POI directly west, sensor 3 should read; no others.
    P.x_position_poi = -1;
    P.y_position_poi = 0;
    
    // sense.
    R.reset_sensors();
    R.sense_poi(P.x_position_poi, P.y_position_poi, P.value_poi);
    
    if(R.sensors.at(0) == 0 && R.sensors.at(1) == 0 && R.sensors.at(2) ==0 && R.sensors.at(3) != 0){
        pass4 = true;
    }
    
    if(VERBOSE){
        cout << "Direct west case: " << endl;
        for(int sen = 0; sen < R.sensors.size(); sen++){
            cout << R.sensors.at(sen) << "\t";
        }
        cout << endl;
    }
    assert(pass4 == true);
    
    
    if(pass1 && pass2 && pass3 && pass4){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

//Test for stationary rovers test in all directions
bool rover_sensor_test(){
    bool passfail = false;
    
    bool pass5 = false;
    bool pass6 = false;
    bool pass7 = false;
    bool pass8 = false;
    
    Rover R1;
    Rover R2;
    R1.x_position = 0;
    R1.y_position = 0;
    R1.theta = 0; // north
    R2.theta = 0;
    
    // case 1, Rover 2 to the north
    R2.x_position = 0;
    R2.y_position = 1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 4 should fire, none other.
    if(R1.sensors.at(4) != 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass5 = true;
    }
    assert(pass5 == true);
    
    // case 2, Rover 2 to the east
    R2.x_position = 1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 5 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) != 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) == 0){
        pass6 = true;
    }
    assert(pass6 == true);
    
    // case 3, Rover 2 to the south
    R2.x_position = 0;
    R2.y_position = -1;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 6 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) != 0 && R1.sensors.at(7) == 0){
        pass7 = true;
    }
    assert(pass7 == true);
    
    // case 4, Rover 2 to the west
    R2.x_position = -1;
    R2.y_position = 0;
    R1.reset_sensors();
    R1.sense_rover(R2.x_position,R2.y_position);
    /// sensor 7 should fire, none other.
    if(R1.sensors.at(4) == 0 && R1.sensors.at(5) == 0 && R1.sensors.at(6) == 0 && R1.sensors.at(7) != 0){
        pass8 = true;
    }
    assert(pass8 == true);
    
    if(pass5 && pass6 && pass7 && pass8){
        passfail = true;
    }
    assert(passfail == true);
    return passfail;
}

void custom_test(){
    Rover R;
    POI P;
    R.x_position = 0;
    R.y_position = 0;
    R.theta = 90;
    
    P.x_position_poi = 0.56;
    P.y_position_poi = -1.91;
    P.value_poi = 100;
    
    R.reset_sensors();
    R.sense_poi(P.x_position_poi,P.y_position_poi,P.value_poi);
    
    
}

//x and y position of poi
vector< vector <double> > poi_positions;
vector<double> poi_positions_loc;

void stationary_rover_test(double x_start,double y_start){//Pass x_position,y_position
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    //x and y position of poi
    vector< vector <double> > poi_positions;
    vector<double> poi_positions_loc;
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    int quad_0=0,quad_1=0,quad_2=0,quad_3=0,quad_0_1=0;
    while (angle<360) {
        if ((0<=angle && 45>= angle)) {
            quad_0++;
        }else if ((45<angle && 135>= angle)) {
            // do something in Q2
            quad_1++;
        }else if((135<angle && 225>= angle)){
            //do something in Q3
            quad_2++;
        }else if((225<angle && 315>= angle)){
            //do something in Q4
            quad_3++;
        }else if ((315<angle && 360> angle)){
            quad_0_1++;
        }
        poi_positions_loc.push_back(R_obj.x_position+(radius*cos(angle * (PI /180))));
        poi_positions_loc.push_back(R_obj.y_position+(radius*sin(angle * (PI /180))));
        poi_positions.push_back(poi_positions_loc);
        poi_positions_loc.clear();
        angle+=7;
    }
    
    vector<bool> checkPass_quad_1,checkPass_quad_2,checkPass_quad_3,checkPass_quad_0;
    
    for (int i=0; i<poi_positions.size(); i++) {
        for (int j=0; j<poi_positions.at(i).size(); j++) {
            P_obj.x_position_poi = poi_positions.at(i).at(j);
            P_obj.y_position_poi = poi_positions.at(i).at(++j);
            R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
            if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
                checkPass_quad_0.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_1.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0){
                checkPass_quad_2.push_back(true);
            }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0){
                checkPass_quad_3.push_back(true);
            }
            R_obj.reset_sensors();
        }
    }
    if (checkPass_quad_0.size() != (quad_0_1+quad_0)) {
        cout<<"Something wrong with quad_0"<<endl;;
    }else if (checkPass_quad_1.size() != (quad_1)){
        cout<<"Something wrong with quad_1"<<endl;
    }else if (checkPass_quad_2.size() != quad_2){
        cout<<"Something wrong with quad_2"<<endl;
    }else if (checkPass_quad_3.size() != quad_3){
        cout<<"Something wrong with quad_3"<<endl;
    }
}

void find_x_y_stationary_rover_test_1(double angle, double radius, double x_position, double y_position){
    poi_positions_loc.push_back(x_position+(radius*cos(angle * (PI /180))));
    poi_positions_loc.push_back(y_position+(radius*sin(angle * (PI /180))));
}

void stationary_rover_test_1(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj;
    
    R_obj.reset_sensors();
    
    R_obj.x_position =x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    int radius = 2;
    
    bool check_pass = false;
    
    double angle=0;
    
    P_obj.value_poi=100;
    
    while (angle<360) {
        find_x_y_stationary_rover_test_1(angle, radius, R_obj.x_position, R_obj.y_position);
        P_obj.x_position_poi = poi_positions_loc.at(0);
        P_obj.y_position_poi = poi_positions_loc.at(1);
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1"<<endl;
                
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2"<<endl;
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3"<<endl;
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<angle<<" with x_position and y_position"<<R_obj.x_position<<R_obj.y_position<<endl;
            exit(10);
        }
        assert(check_pass==true);
        poi_positions_loc.clear();
        R_obj.reset_sensors();
        angle+=7;
        check_pass=false;
    }
}

void stationary_poi_test(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    P_obj.x_position_poi=x_start;
    P_obj.y_position_poi=y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_poi(P_obj.x_position_poi, P_obj.y_position_poi, P_obj.value_poi);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(0) != 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 0"<<endl;
            }
            check_pass = true;
        }else  if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) != 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 1";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) !=0 && R_obj.sensors.at(3) == 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 2";
            }
            check_pass = true;
        }else if (R_obj.sensors.at(0) == 0 && R_obj.sensors.at(1) == 0 && R_obj.sensors.at(2) ==0 && R_obj.sensors.at(3) != 0) {
            if (VERBOSE) {
                cout<<"Pass Quad 3";
            }
            check_pass = true;
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
}

void two_rovers_test(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj; //Rover object
    POI P_obj; // POI object
    vector<double> rover_position_loc;
    
    R_obj.reset_sensors();
    
    double otherRover_x = x_start;
    double otherRover_y = y_start;
    P_obj.value_poi=100;
    R_obj.theta=0.0;
    
    R_obj.x_position =0.0;
    R_obj.y_position =0.0;
    
    bool check_pass = false;
    
    for (int i=0; i<=R_obj.theta; ) {
        if (R_obj.theta > 360) {
            break;
        }
        R_obj.sense_rover(otherRover_x, otherRover_y);
        if (VERBOSE) {
            cout<<endl;
            for (int j=0; j<R_obj.sensors.size(); j++) {
                cout<<R_obj.sensors.at(j)<<"\t";
            }
            cout<<endl;
        }
        if (R_obj.sensors.at(4) != 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if ((0<=R_obj.theta && 45>= R_obj.theta)||(315<R_obj.theta && 360>= R_obj.theta)) {
                if (VERBOSE) {
                    cout<<"Pass Quad 0"<<endl;
                }
                check_pass = true;
            }
            
        }else  if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) != 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) == 0) {
            if((45<R_obj.theta && 135>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 1";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) !=0 && R_obj.sensors.at(7) == 0) {
            if((135<R_obj.theta && 225>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 2";
                }
                check_pass = true;
            }
        }else if (R_obj.sensors.at(4) == 0 && R_obj.sensors.at(5) == 0 && R_obj.sensors.at(6) ==0 && R_obj.sensors.at(7) != 0) {
            if((225<R_obj.theta && 315>= R_obj.theta)){
                if (VERBOSE) {
                    cout<<"Pass Quad 3";
                }
                check_pass = true;
            }
        }else{
            cout<<"Issue at an angle ::"<<R_obj.theta<<" with x_position and y_position"<<P_obj.x_position_poi<<P_obj.y_position_poi<<endl;
            exit(10);
        }
        assert(check_pass==true);
        i+=7;
        R_obj.theta+=7;
        R_obj.reset_sensors();
    }
    
}

vector<double> row_values;
vector< vector <double> > assert_check_values;

void fill_assert_check_values(){
    //First set of x , y thetha values
    for(int i=0;i<3;i++)
        row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //second set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(1);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //third set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(2);
    row_values.push_back(45);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fourth set of x,y,thetha values
    row_values.push_back(1);
    row_values.push_back(3);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //fifth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(4);
    row_values.push_back(315);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
    //sixth set of x,y,thetha values
    row_values.push_back(0);
    row_values.push_back(5);
    row_values.push_back(0);
    assert_check_values.push_back(row_values);
    row_values.clear();
    
}

bool tolerance(double delta_maniplate,double check_value){
    double delta = 0.0000001;
    if (((delta+ delta_maniplate)>check_value)|| ((delta- delta_maniplate)<check_value) || (( delta_maniplate)==check_value)) {
        return true;
    }else{
        return false;
    }
}


void test_path(double x_start, double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    //given
    R_obj.x_position=x_start;
    R_obj.y_position=y_start;
    R_obj.theta=0.0;
    
    P_obj.x_position_poi=1.0;
    P_obj.y_position_poi=1.0;
    P_obj.value_poi=100;
    
    
    
    fill_assert_check_values();
    
    int step_number = 0;
    bool check_assert = false;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==0) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    double dx=0.0,dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==1) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    
    dx=1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==2) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==3) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=-1.0;
    dy=1.0;
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==4) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
    dx=1/sqrt(2.0);
    dy=1/sqrt(2.0);
    R_obj.move_rover(dx, dy);
    R_obj.reset_sensors();
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    if (step_number==5) {
        if(tolerance(R_obj.x_position, assert_check_values.at(step_number).at(0))){
            if(tolerance(R_obj.y_position, assert_check_values.at(step_number).at(1))){
                if(tolerance(R_obj.theta, assert_check_values.at(step_number).at(2))){
                    check_assert=true;
                    step_number++;
                }
            }
        }
    }
    assert(check_assert);
    check_assert=false;
    
}

vector< vector <double> > point_x_y_circle;
vector<double> temp;

void find_x_y_test_circle_path(double start_x_position,double start_y_position,double angle){
    double radius = 1.0;
    temp.push_back(start_x_position+(radius*cos(angle * (PI /180))));
    temp.push_back(start_y_position+(radius*sin(angle * (PI/180))));
}

void test_circle_path(double x_start,double y_start){
    bool VERBOSE = false;
    Rover R_obj;
    POI P_obj;
    
    P_obj.x_position_poi=0.0;
    P_obj.y_position_poi=0.0;
    P_obj.value_poi=100.0;
    
    if (VERBOSE) {
        cout<<R_obj.x_position<<"\t"<<R_obj.y_position<<"\t"<<R_obj.theta<<endl;
    }
    
    double dx=0.0,dy=1.0;
    double angle=0.0;
    
    for(;angle<=360;){
        R_obj.x_position=x_start;
        R_obj.y_position=y_start;
        R_obj.theta=0.0;
        find_x_y_test_circle_path(x_start, y_start,angle);
        dx=temp.at(0);
        dy=temp.at(1);
        R_obj.move_rover(dx, dy);
        assert(tolerance(R_obj.x_position, dx));
        assert(tolerance(R_obj.y_position, dy));
        assert(tolerance(R_obj.theta, angle));
        temp.clear();
        angle+=15.0;
    }
    
}



void obstacle_test(){
    
    Rover R;
    POI P;
    
    R.x_position = 0.0;
    R.y_position = 0.0;
    //R.theta = 0.0;
    //R.phi = 0.0;
    //R.gamma = 0.0;
    
    R.reset_sensors();
    
    R.destination_x_position = 0.707;
    R.destination_y_position = 0.707;
    
    P.x_position_poi = 0.707;
    P.y_position_poi = 0.707;
    
    P.x_position_poi_vec.push_back(0.707);
    P.y_position_poi_vec.push_back(0.707);
    
    double obstacle_x_position = 0.707;
    double obstacle_y_position= 0.707;
    
    //double distance = R.sense_obstacle(obstacle_x_position, obstacle_y_position, R.x_position, R.y_position);
    R.find_gamma(obstacle_x_position, obstacle_y_position);
    cout<<R.gamma<<endl;
    
    R.sense_poi(P.x_position_poi, P.y_position_poi, 100.0);
    cout<<R.phi<<endl;
    cout<<R.theta<<endl;
    
    
    
    
}

void test_all_sensors(){
    POI_sensor_test();
    rover_sensor_test();
    custom_test();
    double x_start = 0.0, y_start = 0.0;
    stationary_rover_test(x_start,y_start);
    stationary_rover_test_1(x_start, y_start);
    stationary_poi_test(x_start,y_start);
    two_rovers_test(x_start,y_start);
    test_path(x_start,y_start);
    x_start = 0.0;
    y_start = 0.0;
    test_circle_path(x_start,y_start);
}

/*******************************************************
 Simulation : All rovers are in simulation
 Reward : Rewards
 *******************************************************/

void simulation( vector<Rover>* teamRover, POI* individualPOI,double scaling_number,vector<vector<double>>* p_ob){
    
    for (int rover_number =0; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.clear();
        for (int poi_number =0; poi_number<individualPOI->value_poi_vec.size(); poi_number++) {
            teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.push_back(99999999.9999);
        }
        
    }
    
    //setting all rovers to inital state
    for (int temp_rover_number =0 ; temp_rover_number<teamRover->size(); temp_rover_number++) {
        teamRover->at(temp_rover_number).x_position = teamRover->at(temp_rover_number).x_position_vec.at(0);
        teamRover->at(temp_rover_number).y_position = teamRover->at(temp_rover_number).y_position_vec.at(0);
        teamRover->at(temp_rover_number).network_for_agent.at(0).store_x_values.push_back(teamRover->at(temp_rover_number).x_position);
        teamRover->at(temp_rover_number).network_for_agent.at(0).store_y_values.push_back(teamRover->at(temp_rover_number).y_position);
        teamRover->at(temp_rover_number).theta = 0.0;
    }
    
    for (int time_step = 1; time_step <100; time_step++) {
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            
            //cout<<"X and Y"<<teamRover->at(rover_number).x_position<<"\t"<<teamRover->at(rover_number).y_position<<endl;
            
            //reset_sense_new(rover_number, p_rover, p_poi); // reset and sense new values
            teamRover->at(rover_number).reset_sensors(); // Reset all sensors
            teamRover->at(rover_number).sense_values_all(rover_number, teamRover, p_ob); // sense all values
            
            //Change of input values
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(rover_number).sensors.size(); change_sensor_values++) {
                teamRover->at(rover_number).sensors.at(change_sensor_values) /= scaling_number;
            }
            
            teamRover->at(rover_number).network_for_agent.at(0).feedForward(teamRover->at(rover_number).sensors); // scaled input into neural network
            
            for (int change_sensor_values = 0 ; change_sensor_values <teamRover->at(rover_number).sensors.size(); change_sensor_values++) {
                assert(!isnan(teamRover->at(rover_number).sensors.at(change_sensor_values)));
            }
            
            double dx = teamRover->at(rover_number).network_for_agent.at(0).outputvaluesNN.at(0);
            double dy = teamRover->at(rover_number).network_for_agent.at(0).outputvaluesNN.at(1);
            teamRover->at(rover_number).network_for_agent.at(0).outputvaluesNN.clear();
            
            assert(!isnan(dx));
            assert(!isnan(dy));
            teamRover->at(rover_number).move_rover(dx, dy);
            
            //cout<<"X and Y"<<teamRover->at(rover_number).x_position<<"\t"<<teamRover->at(rover_number).y_position<<endl;
            
            teamRover->at(rover_number).network_for_agent.at(0).store_x_values.push_back(teamRover->at(rover_number).x_position);
            teamRover->at(rover_number).network_for_agent.at(0).store_y_values.push_back(teamRover->at(rover_number).y_position);
            
//            for (int cal_dis =0; cal_dis<individualPOI->value_poi_vec.size(); cal_dis++) {
//                double x_distance_cal =((teamRover->at(rover_number).x_position) -(individualPOI->x_position_poi_vec.at(cal_dis)));
//                double y_distance_cal = ((teamRover->at(rover_number).y_position) -(individualPOI->y_position_poi_vec.at(cal_dis)));
//                double distance = sqrt((x_distance_cal*x_distance_cal)+(y_distance_cal*y_distance_cal));
//                if (teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(cal_dis) > distance) {
//                    teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(cal_dis) = distance ;
//                }
//            }
        }
    }
}


//This function checks if its colliding with agent
bool check_safe_distance(double safe_distance, double cal_distance_between){
    if (cal_distance_between <= (safe_distance+0.0005)) {
        return true;
    }
    return false;
}

//This is to check if its in between safe distance and
bool check_if_between(double distance_between_rovers, double cal_distance_between){
    if (((distance_between_rovers+0.0005) <= cal_distance_between)) {
        return true;
    }
    return false;
}

bool check_same_quad(double x1, double x2, double y1, double y2){
    //Both are x positve
    if ((x1 <= 0)&&(x2 <= 0) ) {
        if ((y1 <= 0)&&(y2<=0)) {
            //Both y are positive
            return  true;
        }else if((y1 >= 0)&&(y2 >= 0)){
            //Both y are negative
            return true;
        }
    }else if ((x1 >= 0)&&(x2 >= 0)){
        //Both x are negative
        if ((y1 <= 0)&&(y2<=0)) {
            //Both y are positive
            return  true;
        }else if((y1 >= 0)&&(y2 >= 0)){
            //Both y are negative
            return true;
        }
    }
    
    return false;
}

bool check_x_quad(double x1, double x2){
    //Both are positive
    if ((x1<=0)&&(x2<=0)) {
        return true;
    }else if ((x1>=0)&&(x2>=0)){
        return true;
    }
    
    return false;
}


void cal(vector<Rover>* teamRover, vector<vector<double>>* p_location_obstacle, double distance_between_rovers,double radius_of_obstacles, double safe_distance){
    
    bool verbose = false;
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles = 0 ;
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents = 0 ;
        teamRover->at(rover_number).network_for_agent.at(0).path_value = 0 ;
    }
    
    
    //Check distance between each agent
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents = 0;
        for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
            if (rover_number != other_rover) {
                for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
                    if (verbose) {
                        cout<<"Distance values rovers ::"<<endl;
                        cout<<teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index)<<"\t"<< teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index)<<"\t"<< teamRover->at(other_rover).network_for_agent.at(0).store_x_values.at(index)<<"\t"<< teamRover->at(other_rover).network_for_agent.at(0).store_y_values.at(index)<<endl;
                    }
                    
                    double cal_distance_between = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(other_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(other_rover).network_for_agent.at(0).store_y_values.at(index));
                    
                    /*
                     First check if they are colliding or inside safe distance, punish them very high;
                     If they are in formation, give them good score;
                     If not check if they are in formation, if not punish them medium;
                     */
                    if (check_safe_distance(safe_distance, cal_distance_between)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
                    }else if (check_if_between(distance_between_rovers, cal_distance_between)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }
                    
//                    teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents = 0;
                }
            }
        }
    }
    
    if (verbose) {
        for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
            cout<<teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents<<endl;
        }
    }
    
    //check collision with obstacles
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles = 0;
        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
            for (int index_1 = 0 ; index_1 < p_location_obstacle->size(); index_1++) {
                if (verbose) {
                    cout<<"Distance values obstacles ::"<<endl;
                    cout<<teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index)<<"\t"<< teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index)<<"\t"<< p_location_obstacle->at(index_1).at(0)<<"\t"<<p_location_obstacle->at(index_1).at(1) <<endl;
                }
                
                double dist = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), p_location_obstacle->at(index_1).at(0), p_location_obstacle->at(index_1).at(1));
                
                if (radius_of_obstacles >= dist) {
                    teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
                }else{
                    teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 0.1;
                }
            }
        }
    }
    
    if (verbose) {
        for (int rover_number =0 ; rover_number < teamRover->size(); rover_number++) {
            cout<<teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles<<endl;
        }
    }
    
    //Check for path
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        vector<double> distance_all_points;
        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
            double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index),teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index) , teamRover->at(rover_number).destination_x_position, teamRover->at(rover_number).destination_y_position);
            distance_all_points.push_back(distance);
        }
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.clear();
        double temp = *min_element(distance_all_points.begin(), distance_all_points.end());
        
//        //debug
//        for (int i =0 ; i < distance_all_points.size(); i++) {
//            cout<<distance_all_points.at(i)<<"\t";
//        }
//        cout<<endl;
//        cout<<"min"<<temp<<endl;
//        FILE*  p_debug;
//        p_debug = fopen("Debug.txt", "a");
//        for (int t = 0 ; t< teamRover->at(rover_number).network_for_agent.at(0).store_y_values.size(); t++) {
//            fprintf(p_debug, "%f \t %f \t %f \n",teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(t),teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(t) ,distance_all_points.at(t));
//        }
//        fclose(p_debug);
//
//        int index = 0;
//        for ( ; index < distance_all_points.size(); ) {
//            if (temp == distance_all_points.at(index)) {
//                break;
//            }else{
//                index++;
//            }
//        }
        
//        cout<<"Values::"<<endl;
//        cout<<teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index)<<"\t"<<teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index)<<"\t"<<teamRover->at(rover_number).destination_x_position<<"\t"<<teamRover->at(rover_number).destination_y_position<<endl;
        
//        bool check_punishment = check_same_quad(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index),teamRover->at(rover_number).destination_x_position,teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index),teamRover->at(rover_number).destination_y_position);
//
//        bool check_x = check_x_quad(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).destination_x_position);
//
//        bool check_y = check_x_quad(teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(rover_number).destination_y_position);
//
//        if (!check_punishment) {
//            temp = temp +1000;
//        }
//
//        if (!check_x) {
//            temp = temp +1000;
//        }
//
//        if (!check_y) {
//            temp = temp +1000;
//        }
////        cout<<"Finial:"<<temp<<endl;
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.push_back(temp);
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.size() == 1);
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).path_value = (teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(0));
    }

}

void cal_new_way(vector<Rover>* teamRover, vector<vector<double>>* p_location_obstacle, double distance_between_rovers, double radius_of_obstacles, double safe_distance){
    
    //For each rover do all calculations
//    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
//        bool open_area = true;
//        int right_rover, left_rover;
//        if ((rover_number != 0) && (rover_number != (teamRover->size()-1) )) {
//            right_rover = rover_number+1;
//            left_rover = rover_number-1;
//        }else if (rover_number == 0){
//            right_rover = rover_number+1;
//            left_rover = 999999;
//        }else if (rover_number == (teamRover->size()-1)){
//            left_rover = rover_number-1;
//            right_rover = 999999;
//        }
//
//        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
//            //check if its with free world
//            for (int ob = 0 ; ob < p_location_obstacle->size(); ob++) {
//                double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), p_location_obstacle->at(ob).at(0), p_location_obstacle->at(ob).at(0));
//                if (distance < 5) {
//                    open_area = false;
//                    break;
//                }
//            }
//
//            for (int other_rover = 0; other_rover < teamRover->size(); other_rover++) {
//                if (rover_number != other_rover) {
//                    double cal_distance_between = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(other_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(other_rover).network_for_agent.at(0).store_y_values.at(index));
//                    if (open_area) {
//                        //Its open area
//                        if (check_safe_distance(safe_distance, cal_distance_between)) {
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
//                        }else if (check_if_between(distance_between_rovers, cal_distance_between)){
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
//                        }else{
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
//                        }
//                    }else{
//                        //Its near to an obstacle
//                        if (check_safe_distance(safe_distance, cal_distance_between)) {
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
//                        }else if (check_if_between(distance_between_rovers, cal_distance_between)){
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
//                        }else{
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.01;
//                        }
//                    }
//
//                }
//
//                for (int index_1 = 0 ; index_1 < p_location_obstacle->size(); index_1++) {
//                    double dist = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), p_location_obstacle->at(index_1).at(0), p_location_obstacle->at(index_1).at(1));
//
//                    if (open_area) {
//                        //Its open area
//                        if (radius_of_obstacles >= dist) {
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
//                        }else{
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 0.1;
//                        }
//                    }else{
//                        //Its near to an obstacle
//                        if (radius_of_obstacles >= dist) {
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 10;
//                        }else{
//                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
//                        }
//                    }
//                }
//            }
//        }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        bool open_area = true;
        int right_rover=123412312, left_rover=12342134;
        if ((rover_number != 0) && (rover_number != (teamRover->size()-1) )) {
            right_rover = rover_number+1;
            left_rover = rover_number-1;
        }else if (rover_number == 0){
            right_rover = rover_number+1;
            left_rover = 999999;
        }else if (rover_number == (teamRover->size()-1)){
            left_rover = rover_number-1;
            right_rover = 999999;
        }
        
        
        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
            //check if its with free world
            for (int ob = 0 ; ob < p_location_obstacle->size(); ob++) {
                double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), p_location_obstacle->at(ob).at(0), p_location_obstacle->at(ob).at(0));
                if (distance < 5) {
                    open_area = false;
                    break;
                }
            }
            
            for (int index_1 = 0 ; index_1 < p_location_obstacle->size(); index_1++) {
                double dist = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), p_location_obstacle->at(index_1).at(0), p_location_obstacle->at(index_1).at(1));
                
                if (open_area) {
                    //Its open area
                    if (radius_of_obstacles >= dist) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 0.1;
                    }
                }else{
                    //Its near to an obstacle
                    if (radius_of_obstacles >= dist) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 10;
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
                    }
                }
            }
            
            if (right_rover == 999999) {
                double cal_distance_between = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(left_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(left_rover).network_for_agent.at(0).store_y_values.at(index));
                
                if (open_area) {
                    //Its open area
                    if (check_safe_distance(safe_distance, cal_distance_between)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
                    }else if (check_if_between(distance_between_rovers, cal_distance_between)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }
                }else{
                    //Its near to an obstacle
                    if (check_safe_distance(safe_distance, cal_distance_between)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }else if (check_if_between(distance_between_rovers, cal_distance_between)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.01;
                    }
                }
                
            }else if (left_rover == 999999){
                double cal_distance_between = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(right_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(right_rover).network_for_agent.at(0).store_y_values.at(index));
                if (open_area) {
                    //Its open area
                    if (check_safe_distance(safe_distance, cal_distance_between)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
                    }else if (check_if_between(distance_between_rovers, cal_distance_between)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }
                }else{
                    //Its near to an obstacle
                    if (check_safe_distance(safe_distance, cal_distance_between)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }else if (check_if_between(distance_between_rovers, cal_distance_between)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (cal_distance_between*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.01;
                    }
                }
            }else{
                double distance_right = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(right_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(right_rover).network_for_agent.at(0).store_y_values.at(index));
                double distance_left = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), teamRover->at(left_rover).network_for_agent.at(0).store_x_values.at(index), teamRover->at(left_rover).network_for_agent.at(0).store_y_values.at(index));
                if (open_area) {
                    //Its open area
                    //Right
                    if (check_safe_distance(safe_distance, distance_right)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
                    }else if (check_if_between(distance_between_rovers, distance_right)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (distance_right*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }
                    
                    //left
                    //Right
                    if (check_safe_distance(safe_distance, distance_left)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 1;
                    }else if (check_if_between(distance_between_rovers, distance_right)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (distance_left*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }
                }else{
                    //Its near to an obstacle
                    //right
                    if (check_safe_distance(safe_distance, distance_right)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }else if (check_if_between(distance_between_rovers, distance_right)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (distance_right*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.01;
                    }
                    //left
                    if (check_safe_distance(safe_distance, distance_left)) {
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.1;
                    }else if (check_if_between(distance_between_rovers, distance_left)){
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += (distance_left*1);
                    }else{
                        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents += 0.01;
                    }
                }
                
                for (int index_1 = 0 ; index_1 < p_location_obstacle->size(); index_1++) {
                    double dist = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index), p_location_obstacle->at(index_1).at(0), p_location_obstacle->at(index_1).at(1));
                    
                    if (open_area) {
                        //Its open area
                        if (radius_of_obstacles >= dist) {
                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 1;
                        }else{
                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 0.1;
                        }
                    }else{
                        //Its near to an obstacle
                        if (radius_of_obstacles >= dist) {
                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 10;
                        }else{
                            teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles += 0.1;
                        }
                    }
                }
                
            }
        }
        
        
        
        // distance calculations
        vector<double> distance_all_points;
        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
            double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index),teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index) , teamRover->at(rover_number).destination_x_position, teamRover->at(rover_number).destination_y_position);
            distance_all_points.push_back(distance);
        }
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.clear();
        double temp = *min_element(distance_all_points.begin(), distance_all_points.end());
//        double temp= 0.0;
//        for (int i = 0 ; i< distance_all_points.size(); i++) {
//            temp = temp+distance_all_points.at(i);
//        }
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.push_back(temp);
        assert(teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.size() == 1);
        teamRover->at(rover_number).network_for_agent.at(0).path_value = (teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(0));
    }
//    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
//        if (teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(0) < 10) {
//            teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.at(0) = 0.0;
//        }
//    }
    
    
    //test for two objectives
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents = 0.0;
    }
    
}

//Here it checks for domination and we use three objectives

int check_domination(int rover_number,int other_rover, vector<Rover>* teamRover){
    
    int number_of_objectives = 3;
    
    //true means that value is better
    vector<bool> first_policy,second_policy;
    for (int i = 0 ; i< number_of_objectives; i++) {
        first_policy.push_back(false);
        second_policy.push_back(false);
    }
//    cout<<teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents <<"\t"<<teamRover->at(other_rover).network_for_agent.at(0).collision_with_agents<<endl;
//    cout<<teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles <<"\t"<<teamRover->at(other_rover).network_for_agent.at(0).collision_with_obstacles<<endl;
//    cout<<teamRover->at(rover_number).network_for_agent.at(0).path_value <<"\t"<<teamRover->at(other_rover).network_for_agent.at(0).path_value<<endl;
    
    //First check for agent collision
    if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents <= teamRover->at(other_rover).network_for_agent.at(0).collision_with_agents) {
        //rover_number is better
        first_policy.at(0) =true;
    }else if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents == teamRover->at(other_rover).network_for_agent.at(0).collision_with_agents){
        first_policy.at(0) =true;
        second_policy.at(0) =true;
    } else{
        //otherrover is better
        second_policy.at(0) =true;
    }

    //Second check for agent with obstacle
    if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles < teamRover->at(other_rover).network_for_agent.at(0).collision_with_obstacles) {
        //rover_number is better
        first_policy.at(1) =true;
    }else if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles == teamRover->at(other_rover).network_for_agent.at(0).collision_with_obstacles){
        first_policy.at(1) = true;
        second_policy.at(1) = true;
    }else{
        //otherrover is better
        second_policy.at(1) =true;
    }
    
    //Third check for agent path
    if (teamRover->at(rover_number).network_for_agent.at(0).path_value < teamRover->at(other_rover).network_for_agent.at(0).path_value) {
        //rover_number is better
        first_policy.at(2) =true; // modified from 2 to 1
    }else if (teamRover->at(rover_number).network_for_agent.at(0).path_value == teamRover->at(other_rover).network_for_agent.at(0).path_value){
        first_policy.at(2) =true;
        second_policy.at(2) =true;
    } else{
        //otherrover is better
        second_policy.at(2) =true; // modified from 2 to 1
    }
    
    bool first_all = true;
    bool second_all = true;
    for (int i = 0 ; i < first_policy.size(); i++) {
        if ((first_policy.at(i) == true) && (second_policy.at(i) == false)) {
            first_all = false;
        }
        if ((second_policy.at(i) == true) && (first_policy.at(i) == false)) {
            second_all = false;
        }
    }
    
    if (first_all) {
        return 1;
    }else if (second_all){
        return -1;
    }
    
    return 0;
}

void nsgaii(vector<Rover>* teamRover, int number_of_rovers){
    
    //Set all rovers to false
    for (int rover_number =0 ; rover_number< teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).already_has_front = false;
        teamRover->at(rover_number).network_for_agent.at(0).safe = false;
        teamRover->at(rover_number).network_for_agent.at(0).dominating_me = 0;
    }
    
    //First sort with domination. We need all low domination for simulation
    vector<vector<int>> finial_front;
    vector<int> temp_front;
    bool next_front =false;
    do{
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
                if ((rover_number != other_rover) && (!teamRover->at(rover_number).network_for_agent.at(0).already_has_front) ) {
                    if (!teamRover->at(other_rover).network_for_agent.at(0).already_has_front) {
                        int temp = check_domination(rover_number,other_rover,teamRover);
                        if (temp == -1) {
                            teamRover->at(rover_number).network_for_agent.at(0).dominating_over.push_back(other_rover);
                        }else if (temp == 1){
                            teamRover->at(rover_number).network_for_agent.at(0).dominating_me +=1;
                        }else{
                            //Next level
                            //next_fron = true;
                        }
                    }
                }
            }
            
            if ((teamRover->at(rover_number).network_for_agent.at(0).dominating_me == 0) && (teamRover->at(rover_number).network_for_agent.at(0).already_has_front == false)) {
                temp_front.push_back(rover_number);
                teamRover->at(rover_number).network_for_agent.at(0).rank = 1;
                teamRover->at(rover_number).network_for_agent.at(0).already_has_front = true;
            }
            teamRover->at(rover_number).network_for_agent.at(0).dominating_me = 0;
        }
        
        if (temp_front.size() != 0 ) {
            finial_front.push_back(temp_front);
        }
        temp_front.clear();
        
        for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
            if (teamRover->at(rover_number).network_for_agent.at(0).already_has_front == false) {
                next_front = true;
                break;
            }else{
                next_front = false;
            }
        }
    }while (next_front);
    
    FILE* p_front;
    p_front =fopen("Front", "a");
    for (int i=0; i< finial_front.size(); i++) {
        for (int j =0 ; j< finial_front.at(i).size(); j++) {
            fprintf(p_front, "%d \t",finial_front.at(i).at(j));
        }
        fprintf(p_front, "\n");
    }
    fclose(p_front);
    
    
    //Now remove the agents which didn't perform well. This will end only when rovers will be number_of_rovers
    vector<int> safe_rover;
    for (int front = 0 ; front < finial_front.size(); front++) {
        if (safe_rover.size() < number_of_rovers/2) {
            for (int i =0 ; i <finial_front.at(front).size(); i++) {
                if (safe_rover.size() < number_of_rovers/2) {
                    safe_rover.push_back(finial_front.at(front).at(i));
                }
            }
        }
    }
    assert(safe_rover.size() == (number_of_rovers/2));
    
    //Now remove agents which are not useful
    int count = 0;
    for (int rover_number = 0 ; rover_number <teamRover->size(); rover_number++) {
        for (int index = 0 ; index < safe_rover.size(); index++) {
            if (rover_number == safe_rover.at(index)) {
                teamRover->at(rover_number).network_for_agent.at(0).safe = true;
                count++;
            }
        }
    }
    
    assert(count == safe_rover.size());
    
    vector<int> unsafe;
    for (int index = 0; index < number_of_rovers; index++) {
        if (!(find(safe_rover.begin(), safe_rover.end(), index) != safe_rover.end())) {
            unsafe.push_back(index);
        }
    }
    assert(safe_rover.size() == unsafe.size());
    
//    cout<<"Rover size before::"<<teamRover->size()<<endl;
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); ) {
        if (teamRover->at(rover_number).network_for_agent.at(0).safe) {
            rover_number++;
        }else{
            teamRover->erase(teamRover->begin()+rover_number);
            rover_number = 0;
        }
    }
    
//    cout<<"Rover size outside::"<<teamRover->size()<<endl;
    assert(teamRover->size() == (number_of_rovers/2));
    
    for (int rover_number = 0; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.at(0).safe == true);
    }
    
    for (int index = teamRover->size(); index < number_of_rovers; index++) {
        int num = rand()%teamRover->size();
        teamRover->push_back(teamRover->at(num));
        teamRover->at(num).network_for_agent.at(0).mutate();
    }
    
    assert(teamRover->size() == number_of_rovers);
    
    finial_front.clear();
    safe_rover.clear();
    
    
}

void sort_remove(vector<Rover>* teamRover, int number_of_rovers){
    //create a 3d point for rover
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).ranked = false;
        teamRover->at(rover_number).network_for_agent.at(0).threeD.push_back(teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents);
        teamRover->at(rover_number).network_for_agent.at(0).threeD.push_back(teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles);
        teamRover->at(rover_number).network_for_agent.at(0).threeD.push_back(teamRover->at(rover_number).network_for_agent.at(0).path_value);
    }
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        assert(teamRover->at(rover_number).network_for_agent.at(0).ranked == false);
        assert(teamRover->at(rover_number).network_for_agent.at(0).threeD.size() == 3);
    }
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        for (int other_rover = 0 ; other_rover < teamRover->size(); other_rover++) {
            if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents <= teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents) {
                //Select rover
            }
            
            if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles <= teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles) {
                //Select rover
            }
            
            if (teamRover->at(rover_number).network_for_agent.at(0).path_value <= teamRover->at(rover_number).network_for_agent.at(0).path_value) {
                //Select rover
            }
        }
    }
    
}

void EA_working(vector<Rover>* teamRover, int number_of_rovers){
//    cout<<"EA working::"<<endl;
//    for (int i = 0 ; i < teamRover->size(); i++) {
//        cout<<teamRover->at(i).network_for_agent.at(0).path_value<<"\t";
//    }
//    cout<<endl;
    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).summation = teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles+teamRover->at(rover_number).network_for_agent.at(0).path_value;
        //+teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents
    }
    
    
    for (int half = 0 ; half < (number_of_rovers/2); half++) {
        int num_1 = rand()%teamRover->size();
        int num_2 = rand()%teamRover->size();
        while (num_1 == num_2) {
            num_1 = rand()%teamRover->size();
            num_2 = rand()%teamRover->size();
        }
//        cout<<num_1<<"\t"<<num_2<<endl;
        if (teamRover->at(num_1).network_for_agent.at(0).summation < teamRover->at(num_2).network_for_agent.at(0).summation) {
            //Kill num_2
            teamRover->erase(teamRover->begin()+(num_2));
        }else{
            //Kill num_1
            teamRover->erase(teamRover->begin()+(num_1));
        }
    }
    
//    for (int i = 0 ; i < teamRover->size(); i++) {
//        cout<<teamRover->at(i).network_for_agent.at(0).path_value<<"\t";
//    }
//    cout<<endl;
    
    for (int index = teamRover->size(); index < number_of_rovers; index++) {
        int num = rand()%teamRover->size();
        teamRover->push_back(teamRover->at(num));
        teamRover->at(num).network_for_agent.at(0).mutate();
    }
    
//        cout<<teamRover->size()<<endl;
    
}

void clean(vector<Rover>* teamRover){
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        teamRover->at(rover_number).network_for_agent.at(0).store_x_values.clear();
        teamRover->at(rover_number).network_for_agent.at(0).store_y_values.clear();
        teamRover->at(rover_number).network_for_agent.at(0).closest_dist_to_poi.clear();
        teamRover->at(rover_number).network_for_agent.at(0).crowding_distance.clear();
        teamRover->at(rover_number).network_for_agent.at(0).dominate.clear();
        teamRover->at(rover_number).network_for_agent.at(0).dominating_over.clear();
        teamRover->at(rover_number).network_for_agent.at(0).front.clear();
        teamRover->at(rover_number).x_position_vec.clear();
        teamRover->at(rover_number).y_position_vec.clear();
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles = 0;
        teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents = 0;
    }
}


/***************************
 Main
 **************************/

int main(int argc, const char * argv[]) {
    srand((unsigned)time(NULL));
    vector<vector<double>> weights_standard;
    vector<double> one = {0.671454,0.634757,0.854197,0.991215,0.848234,0.770385,0.86142,0.882041,0.96162,0.942041,0.875442,0.556634,0.841049,0.514911,0.617213,0.506047,0.623957,0.844548,0.821718,0.612752,0.515858,0.527021,0.638291,0.758563,0.659953,0.835606,0.528321,0.999274,0.79068,0.959359,0.940651,0.515921,0.589607,0.523787,0.793157,0.588365,0.643429,0.617049,0.746817,0.753289,0.532379,0.695133,0.592982,0.750443,0.688829,0.648603,0.574555,0.538678,0.568925,0.929559,0.596192,0.702967,0.765885,0.724139,0.609243,0.551718,0.727328,0.694687,0.611897,0.646437,0.662239,0.7432,0.962123,0.893733,0.967089,0.866269,0.889188,0.588209,0.53305,0.963574,0.792271,0.693563,0.712698,0.814482,0.996346,0.588805,0.543164,0.951516,0.633886,0.719896,0.795762,0.868816,0.689555,0.854433,0.949342,0.598054,0.995045,0.725764,0.916608,0.927911,0.903856,0.614513,0.619162,0.762975,0.814336,0.551764,0.997603,0.709565,0.66551,0.721746,0.880128,0.815798,0.616773,0.598599,0.659,0.808434,0.855406,0.807256,0.557897,0.567978,0.513123,0.551441,0.570654,0.973818,0.956401,0.735243,0.721513,0.965253,0.509564,0.744926,0.964209,0.96257,0.920086,0.885662,0.814043,0.626955,0.726342,0.637929,0.671471,0.909083,0.950978,0.591166,0.723333,0.561207,0.709339,0.866406,0.688243,0.80001,0.763484,0.879703,0.664015,0.596741,0.921546,0.916547,0.910284,0.637184,0.654692,0.906434,0.936546,0.536654,0.552018,0.77336,0.854748,0.754369,0.672369,0.506642,0.631187,0.860815,0.722772,0.626511,0.774195,0.894895,0.995559,0.862542,0.745407,0.547812,0.568965,0.600208,0.700796,0.778631,0.954339,0.58148,0.940879,0.851926,0.823781,0.783,0.880843,0.822959,0.970577,0.994674,0.980846,0.574701,0.506356,0.82814,0.554463,0.858738,0.811225,0.754212,0.532766,0.691066,0.746368,0.701496,0.550226,0.652259,0.512962,0.852404,0.857114,0.522182,0.807981,0.728296,0.971199,0.947205,0.668414,0.529575,0.565997,0.710368,0.652641,0.937057,0.620432,0.606916,0.934018,0.541164,0.840332,0.962393,0.936158,0.999327,0.695271,0.925757,0.692803,0.944794,0.658825,0.873941,0.827511,0.969502,0.920257,0.758694,0.875688,0.696114,0.584496,0.626098,0.822415,0.823039,0.812514,0.924211,0.714008,0.831596,0.641568,0.836585,0.989015,0.879119,0.850507,0.969575,0.641229,0.640178,0.972991,0.55657,0.769287,0.901359,0.635379,0.810356,0.646548,0.526174,0.914362,0.681262,0.967802,0.845118,0.905044,0.582031,0.688043,0.94436,0.858364,0.530548,0.916151,0.750493,0.542897,0.97281,0.51655,0.651625,0.859242,0.788685,0.934026,0.672925,0.874846};
    vector<double> two = {0.844764,0.955795,0.540454,0.902153,0.982653,0.945393,0.719744,0.739435,0.681075,0.823167,0.961842,0.686586,0.959024,0.812051,0.643293,0.831569,0.68731,0.620286,0.644904,0.908162,0.974513,0.640482,0.577615,0.974282,0.751003,0.611504,0.541705,0.936291,0.74476,0.675916,0.626326,0.667553,0.561811,0.863138,0.756601,0.687968,0.684115,0.92026,0.805739,0.549333,0.639706,0.530439,0.589962,0.986084,0.608227,0.968327,0.674906,0.653005,0.562584,0.853381,0.769281,0.801972,0.737007,0.878369,0.748522,0.915027,0.851655,0.761681,0.564762,0.960061,0.739959,0.989149,0.619534,0.51543,0.827133,0.621244,0.753396,0.824465,0.78363,0.977461,0.686975,0.997235,0.532568,0.874298,0.820979,0.699582,0.876218,0.593647,0.923551,0.62632,0.563294,0.781167,0.566838,0.848868,0.929215,0.819314,0.715405,0.809413,0.802263,0.628292,0.707514,0.684761,0.781225,0.555856,0.779156,0.773781,0.936485,0.511197,0.681789,0.830414,0.762563,0.894319,0.826021,0.938758,0.702795,0.869944,0.651388,0.883306,0.724628,0.829455,0.650205,0.998349,0.747875,0.528613,0.906419,0.681152,0.629783,0.761342,0.87293,0.827873,0.568721,0.98893,0.953769,0.991224,0.994889,0.591228,0.77175,0.797538,0.720498,0.908754,0.920983,0.963131,0.83675,0.757742,0.870892,0.589735,0.668095,0.674041,0.612566,0.899824,0.848592,0.779483,0.772463,0.78183,0.720886,0.926076,0.567144,0.98353,0.688958,0.814257,0.724756,0.976396,0.793555,0.783741,0.840039,0.536331,0.618851,0.530839,0.817211,0.870336,0.730183,0.693024,0.647031,0.649124,0.826987,0.667141,0.646475,0.796992,0.549738,0.953977,0.983357,0.787496,0.941597,0.919264,0.571639,0.528397,0.763553,0.536514,0.698711,0.741612,0.76822,0.975526,0.658135,0.778385,0.80906,0.868533,0.926149,0.786977,0.726266,0.845686,0.941119,0.891914,0.893096,0.761528,0.996242,0.836446,0.640478,0.519653,0.805423,0.737987,0.848627,0.867545,0.824557,0.832332,0.503823,0.757145,0.831966,0.845425,0.565486,0.627247,0.646918,0.747246,0.960628,0.7734,0.536566,0.563396,0.996828,0.688835,0.747376,0.64721,0.655647,0.962915,0.718479,0.970603,0.929546,0.878044,0.784429,0.900936,0.539687,0.521651,0.892838,0.929218,0.859002,0.743529,0.984002,0.617082,0.790953,0.549548,0.74801,0.804205,0.76793,0.606831,0.512816,0.896908,0.836467,0.996962,0.934365,0.872243,0.792846,0.863524,0.739523,0.659781,0.941321,0.775813,0.583862,0.968093,0.741757,0.706036,0.85231,0.770932,0.561352,0.643923,0.918084,0.736042,0.660473,0.565479,0.99911,0.544481,0.599108,0.711343,0.541806,0.627288,0.874846};
    vector<double> three = {0.825417,0.784251,0.90843,0.987937,0.759446,0.511096,0.982786,0.686599,0.669984,0.913131,0.984753,0.745875,0.928195,0.665153,0.732534,0.706605,0.909132,0.787492,0.876114,0.842591,0.921908,0.502743,0.600078,0.505002,0.576407,0.680526,0.595192,0.900065,0.896253,0.821599,0.622009,0.603684,0.612649,0.78546,0.719369,0.934962,0.902369,0.622912,0.787948,0.534243,0.528671,0.874294,0.762105,0.706571,0.835182,0.89803,0.682008,0.505841,0.670168,0.521334,0.559625,0.621559,0.534837,0.509655,0.778589,0.740711,0.637113,0.958588,0.986272,0.775155,0.536993,0.735125,0.740913,0.526052,0.85969,0.802665,0.891355,0.508684,0.956672,0.788925,0.970859,0.732511,0.814347,0.735191,0.848743,0.831502,0.548652,0.687489,0.626578,0.900468,0.669142,0.768463,0.556402,0.946624,0.916918,0.642073,0.824451,0.548319,0.596554,0.779556,0.993044,0.586425,0.537922,0.854026,0.613572,0.810832,0.649691,0.858338,0.593969,0.843715,0.818805,0.662588,0.618005,0.816259,0.861799,0.748741,0.597279,0.960212,0.776536,0.741281,0.715892,0.504412,0.651765,0.716768,0.719301,0.791823,0.664189,0.520539,0.703053,0.719629,0.802518,0.920038,0.584398,0.975293,0.754068,0.625836,0.928163,0.629974,0.981275,0.781265,0.725136,0.864647,0.618514,0.861167,0.634271,0.695567,0.897814,0.557691,0.605761,0.523711,0.502416,0.613268,0.702482,0.619992,0.708489,0.566362,0.840184,0.968297,0.671139,0.828658,0.759205,0.961689,0.607743,0.837066,0.569151,0.714274,0.797471,0.602701,0.589931,0.968643,0.987364,0.62245,0.521434,0.747104,0.57598,0.99858,0.628525,0.6201,0.516192,0.638516,0.535083,0.641168,0.616567,0.64568,0.93648,0.918532,0.774378,0.963399,0.845001,0.930228,0.849255,0.930502,0.944547,0.500499,0.8875,0.718709,0.834783,0.699757,0.816417,0.51259,0.597586,0.619906,0.759894,0.539848,0.717691,0.727404,0.984015,0.844422,0.700623,0.872947,0.617042,0.626917,0.594375,0.658843,0.675112,0.598905,0.795708,0.965309,0.941141,0.753617,0.543513,0.817098,0.970739,0.70709,0.558234,0.733636,0.717177,0.600679,0.615505,0.799558,0.671609,0.726768,0.788483,0.54007,0.956101,0.68536,0.84486,0.564963,0.827767,0.779041,0.848302,0.917926,0.589606,0.512408,0.537213,0.938226,0.766138,0.973486,0.88016,0.850417,0.963081,0.998158,0.549511,0.638422,0.957424,0.92499,0.806706,0.805461,0.884844,0.578354,0.90303,0.716611,0.577271,0.699532,0.54172,0.679679,0.872664,0.868029,0.957454,0.929486,0.864067,0.880135,0.930998,0.787245,0.731139,0.759617,0.875785,0.825149,0.780485,0.607097,0.971424,0.725204,0.874846};
    vector<double> four={0.510271,0.619942,0.86823,0.840582,0.66378,0.643509,0.948612,0.823016,0.931597,0.848524,0.647371,0.871026,0.827356,0.874028,0.780499,0.854484,0.815138,0.531636,0.711616,0.625933,0.55863,0.897378,0.738047,0.863249,0.623706,0.631935,0.92341,0.747632,0.954499,0.766579,0.885617,0.570484,0.616317,0.931497,0.664845,0.554546,0.753667,0.883126,0.69101,0.809704,0.693243,0.827463,0.675145,0.663605,0.707505,0.533924,0.653435,0.774482,0.715163,0.750814,0.936087,0.820308,0.922101,0.750784,0.926379,0.65878,0.614282,0.743642,0.894508,0.503852,0.739562,0.821227,0.857486,0.759618,0.899761,0.791163,0.577389,0.681776,0.603856,0.506672,0.631452,0.822002,0.892769,0.772269,0.522394,0.875564,0.61095,0.734677,0.711316,0.585067,0.71351,0.954419,0.921202,0.645222,0.748782,0.782688,0.645084,0.923838,0.943885,0.872243,0.793073,0.678119,0.652077,0.960635,0.891571,0.625629,0.948156,0.65517,0.941353,0.81213,0.97387,0.834007,0.659347,0.646514,0.968471,0.597348,0.623805,0.788726,0.618164,0.982906,0.706201,0.627752,0.622128,0.609946,0.864257,0.565186,0.575412,0.951587,0.823427,0.845623,0.88209,0.790342,0.78112,0.789601,0.831128,0.764668,0.774713,0.601057,0.972619,0.801426,0.57326,0.78145,0.833265,0.68991,0.815946,0.610634,0.917484,0.661509,0.984997,0.836456,0.823541,0.754286,0.791711,0.793898,0.540693,0.932393,0.721653,0.817492,0.586404,0.691327,0.637309,0.754076,0.760414,0.786199,0.641292,0.699618,0.980296,0.837825,0.820345,0.536753,0.702514,0.645963,0.705541,0.52204,0.9223,0.590487,0.823256,0.962091,0.862502,0.576643,0.643815,0.594184,0.946435,0.740359,0.716159,0.991884,0.591424,0.563,0.843968,0.571259,0.654007,0.892017,0.636114,0.673863,0.613153,0.765976,0.76469,0.638999,0.658944,0.877946,0.630467,0.757093,0.955737,0.571972,0.627021,0.846452,0.826168,0.905183,0.915963,0.597997,0.529913,0.748806,0.679485,0.610943,0.613673,0.505573,0.67118,0.52382,0.840161,0.593222,0.788411,0.821395,0.680958,0.85822,0.598218,0.747356,0.817027,0.773672,0.60778,0.957388,0.820008,0.869913,0.636253,0.500322,0.904519,0.748447,0.644255,0.995225,0.747713,0.813517,0.777338,0.727395,0.831291,0.516024,0.818366,0.774272,0.697171,0.857355,0.565734,0.792758,0.885048,0.501734,0.636168,0.574224,0.983377,0.613452,0.793166,0.747526,0.661658,0.984684,0.585225,0.874213,0.895872,0.91797,0.825297,0.7618,0.570907,0.738602,0.685605,0.96544,0.644651,0.6521,0.844174,0.532416,0.820049,0.556521,0.946407,0.755238,0.780593,0.918185,0.933672,0.719406,0.874846};
    vector<double> five = {0.549324,0.987201,0.885099,0.864653,0.729613,0.605839,0.829622,0.963515,0.795133,0.79282,0.92761,0.847799,0.954545,0.546273,0.706223,0.995102,0.677059,0.834757,0.754824,0.827392,0.97222,0.608261,0.536075,0.820672,0.529893,0.919104,0.887084,0.725535,0.559566,0.620163,0.577538,0.675273,0.806234,0.873534,0.980784,0.529376,0.727452,0.781084,0.678918,0.569057,0.636982,0.763569,0.806289,0.806269,0.957068,0.938268,0.962199,0.681288,0.906305,0.77585,0.715642,0.790272,0.603588,0.508611,0.721166,0.641769,0.714474,0.65876,0.780797,0.852721,0.673927,0.692406,0.761192,0.860818,0.762866,0.9946,0.740261,0.559521,0.862116,0.590454,0.754975,0.871756,0.604452,0.531914,0.881939,0.748705,0.989807,0.694403,0.824835,0.506017,0.622075,0.720808,0.619222,0.755742,0.751225,0.831279,0.808187,0.700864,0.919336,0.784268,0.690598,0.885582,0.97593,0.953627,0.60638,0.931623,0.793054,0.8622,0.997721,0.696199,0.520345,0.943324,0.941641,0.652192,0.888191,0.833412,0.660433,0.889147,0.898072,0.899353,0.930658,0.564988,0.745801,0.672688,0.873179,0.526748,0.552442,0.888907,0.854821,0.978781,0.864711,0.692483,0.568376,0.696807,0.730595,0.615584,0.622555,0.779699,0.89505,0.612059,0.878567,0.583315,0.771941,0.512626,0.706943,0.586322,0.812478,0.809794,0.708412,0.780172,0.854311,0.911845,0.879133,0.594478,0.897044,0.625938,0.640741,0.926083,0.671285,0.791108,0.651178,0.842238,0.992001,0.556809,0.792741,0.597096,0.897663,0.527351,0.689245,0.644794,0.557589,0.904124,0.61743,0.648646,0.79631,0.585039,0.747344,0.603375,0.923594,0.838089,0.767437,0.807194,0.503086,0.865275,0.672416,0.791936,0.56708,0.921042,0.947684,0.72473,0.529472,0.828163,0.941817,0.624256,0.863239,0.954516,0.54803,0.747555,0.64908,0.592361,0.806888,0.875104,0.873034,0.586615,0.740472,0.616934,0.811415,0.944815,0.50324,0.957384,0.750924,0.785858,0.908074,0.995607,0.668114,0.995709,0.881428,0.656369,0.598339,0.779324,0.6045,0.839079,0.900181,0.848005,0.912513,0.610666,0.961558,0.905465,0.655785,0.776327,0.736351,0.859002,0.739341,0.611285,0.868234,0.913133,0.519076,0.604845,0.635335,0.575307,0.683254,0.952117,0.723297,0.948708,0.930501,0.938286,0.770269,0.903243,0.798603,0.620993,0.532408,0.680724,0.920816,0.660461,0.865589,0.958232,0.50675,0.945414,0.569716,0.714707,0.584267,0.781116,0.714398,0.884522,0.653127,0.605595,0.740673,0.993797,0.749709,0.862196,0.922287,0.869228,0.618447,0.744027,0.864623,0.714074,0.948658,0.595839,0.762025,0.85167,0.513298,0.504651,0.874846};
    vector<double> six = {0.671169,0.8417,0.955831,0.657375,0.999841,0.82778,0.500692,0.63,0.916256,0.519992,0.513339,0.680329,0.788802,0.893784,0.826996,0.817666,0.512754,0.854154,0.767726,0.666254,0.725388,0.596599,0.546696,0.827038,0.520106,0.923853,0.698637,0.993016,0.624113,0.969239,0.992039,0.69163,0.717351,0.511596,0.885814,0.876279,0.625726,0.574749,0.811369,0.673491,0.862299,0.652658,0.716498,0.677158,0.994945,0.536978,0.995376,0.77808,0.690415,0.800873,0.777498,0.905121,0.867433,0.954368,0.562477,0.556032,0.72162,0.772914,0.85826,0.783996,0.614505,0.977779,0.532832,0.804191,0.537131,0.557459,0.711864,0.804518,0.530635,0.890557,0.588633,0.651743,0.841838,0.770022,0.763258,0.570969,0.773264,0.744273,0.500245,0.609392,0.54642,0.686525,0.924769,0.593348,0.899965,0.716636,0.996886,0.663813,0.698872,0.935828,0.954482,0.974283,0.778075,0.600713,0.680514,0.907097,0.586634,0.555742,0.853654,0.860609,0.748079,0.965865,0.790287,0.849199,0.993247,0.50519,0.729534,0.782302,0.645425,0.658465,0.824698,0.693383,0.692212,0.515139,0.945114,0.532841,0.953859,0.500022,0.875838,0.707662,0.678782,0.796635,0.548275,0.858046,0.67656,0.941433,0.672001,0.822249,0.536765,0.913019,0.617851,0.721892,0.834638,0.762684,0.935722,0.682298,0.886381,0.89793,0.511465,0.699193,0.836343,0.92135,0.626416,0.673757,0.837984,0.996408,0.621115,0.574726,0.926922,0.774262,0.517662,0.840877,0.624262,0.964007,0.569307,0.844356,0.597328,0.793646,0.801364,0.529845,0.597766,0.653908,0.724997,0.521471,0.862433,0.905728,0.578024,0.848522,0.612223,0.634819,0.903854,0.576285,0.622836,0.508778,0.537767,0.750193,0.991457,0.920137,0.7508,0.698451,0.872678,0.5999,0.527503,0.748366,0.791526,0.675078,0.534812,0.57809,0.956106,0.774232,0.522213,0.841199,0.52934,0.617406,0.739343,0.633066,0.937899,0.762744,0.942393,0.796067,0.998751,0.500474,0.963761,0.934014,0.973626,0.734989,0.966265,0.514844,0.984452,0.686973,0.954241,0.929729,0.95646,0.72648,0.942763,0.523928,0.654048,0.580229,0.911515,0.82787,0.512504,0.652998,0.940739,0.507326,0.627822,0.807393,0.850271,0.512277,0.834415,0.509682,0.725561,0.99533,0.512669,0.926272,0.855968,0.753246,0.802069,0.87797,0.537947,0.772032,0.546837,0.69109,0.653237,0.956489,0.704111,0.996667,0.983333,0.883624,0.565578,0.662218,0.890231,0.607877,0.593702,0.846139,0.563862,0.829885,0.876018,0.734319,0.695671,0.642664,0.749868,0.534318,0.777372,0.794161,0.969064,0.564444,0.614294,0.945688,0.674047,0.71085,0.751368,0.740228,0.874846};
    vector<double> seven = {0.50784,0.76343,0.971292,0.507176,0.609191,0.671408,0.856917,0.708047,0.645058,0.987573,0.63296,0.664768,0.758769,0.634004,0.710154,0.552593,0.935658,0.606079,0.862057,0.598864,0.614524,0.810842,0.820713,0.722085,0.58147,0.766398,0.848477,0.858728,0.636959,0.868052,0.846859,0.657904,0.884676,0.746774,0.531279,0.711041,0.959541,0.513037,0.608106,0.937533,0.618101,0.928397,0.567703,0.87785,0.520653,0.617912,0.741409,0.867686,0.701191,0.912457,0.672014,0.542706,0.758951,0.681538,0.604092,0.975819,0.586478,0.938956,0.536499,0.943106,0.78322,0.5762,0.691542,0.746545,0.680839,0.855417,0.986312,0.945183,0.684342,0.740166,0.966682,0.528398,0.78407,0.862826,0.518063,0.592933,0.917533,0.977033,0.985597,0.92224,0.584588,0.665959,0.776845,0.936997,0.60192,0.976474,0.590796,0.500971,0.811206,0.942323,0.62037,0.563371,0.57996,0.894318,0.807692,0.877258,0.57513,0.711335,0.910518,0.580892,0.546137,0.928288,0.738023,0.947586,0.586009,0.55547,0.785578,0.71327,0.922634,0.703489,0.541072,0.798721,0.609267,0.94441,0.701617,0.569531,0.611591,0.508958,0.548892,0.730341,0.838724,0.930544,0.650489,0.774758,0.865365,0.689933,0.705981,0.927843,0.764321,0.949484,0.97304,0.886466,0.839035,0.656316,0.707393,0.64906,0.750227,0.566711,0.710642,0.753192,0.900448,0.830162,0.538014,0.900884,0.662786,0.935948,0.972315,0.704511,0.72027,0.570976,0.8935,0.559312,0.864809,0.851368,0.949489,0.556505,0.671208,0.991993,0.930648,0.903596,0.739303,0.957537,0.819484,0.567162,0.785581,0.758353,0.646062,0.867948,0.595157,0.804392,0.923659,0.944922,0.801818,0.656477,0.910018,0.672675,0.646903,0.506217,0.991366,0.896293,0.988453,0.930785,0.710767,0.855789,0.750691,0.855799,0.918531,0.749435,0.752468,0.730839,0.712258,0.920069,0.607575,0.513355,0.958646,0.969855,0.852449,0.613662,0.819779,0.529858,0.831274,0.715355,0.96539,0.806807,0.505728,0.772012,0.701193,0.956215,0.599788,0.632062,0.571838,0.883162,0.804903,0.502268,0.61911,0.885161,0.90448,0.603221,0.840156,0.50345,0.984212,0.650546,0.73462,0.751623,0.526871,0.622086,0.899292,0.904249,0.712917,0.996808,0.854512,0.781911,0.576859,0.768637,0.978451,0.829437,0.84221,0.526003,0.533828,0.540647,0.650941,0.870813,0.759479,0.558707,0.681823,0.890993,0.919714,0.636732,0.559021,0.957672,0.592434,0.53467,0.694463,0.846967,0.972686,0.932759,0.886687,0.553195,0.544435,0.825147,0.742167,0.598086,0.527767,0.674158,0.581876,0.595766,0.537789,0.618928,0.828577,0.88709,0.822441,0.760458,0.874846};
    vector<double> eight = {0.525262,0.583471,0.899206,0.958879,0.879972,0.683708,0.577473,0.594889,0.803807,0.592437,0.581972,0.708855,0.725534,0.554083,0.977942,0.776009,0.890855,0.602784,0.992624,0.52603,0.984642,0.880319,0.52871,0.535712,0.717888,0.545757,0.545806,0.856414,0.751283,0.816924,0.545688,0.879536,0.858301,0.95738,0.691565,0.637503,0.519508,0.863681,0.895002,0.79976,0.571536,0.806924,0.974749,0.601055,0.932994,0.825605,0.943514,0.635832,0.932742,0.59955,0.629054,0.518572,0.639782,0.811377,0.813978,0.531365,0.645994,0.715691,0.615453,0.921819,0.511362,0.956476,0.988213,0.895243,0.854915,0.561085,0.648566,0.954103,0.617448,0.945798,0.52305,0.907912,0.775552,0.709845,0.86598,0.528078,0.903007,0.83313,0.922959,0.672153,0.87323,0.884197,0.700973,0.752401,0.604625,0.924544,0.814206,0.853591,0.807891,0.731734,0.749582,0.719692,0.860862,0.512053,0.580908,0.815333,0.802049,0.538685,0.677524,0.647907,0.875643,0.932198,0.944071,0.501809,0.900338,0.978529,0.644255,0.993312,0.586724,0.568016,0.637381,0.965806,0.79577,0.513616,0.835943,0.686301,0.664758,0.591097,0.568431,0.621731,0.932167,0.930364,0.624471,0.9782,0.60117,0.868326,0.950239,0.672205,0.74573,0.979895,0.593502,0.986408,0.561786,0.932554,0.94211,0.549957,0.629807,0.665159,0.828184,0.795683,0.539578,0.680079,0.590906,0.852286,0.877685,0.74788,0.621102,0.85423,0.547048,0.727717,0.73817,0.923114,0.777626,0.552512,0.574206,0.687067,0.527024,0.697865,0.524206,0.827666,0.58041,0.952533,0.7166,0.894299,0.977583,0.744892,0.898245,0.807421,0.821211,0.601372,0.75127,0.593395,0.694921,0.535929,0.851883,0.592313,0.998245,0.99622,0.964329,0.979034,0.631201,0.596925,0.520533,0.605674,0.5607,0.683297,0.665772,0.628046,0.575617,0.901431,0.851597,0.782633,0.713617,0.756675,0.943056,0.938817,0.701162,0.92212,0.566511,0.85806,0.910885,0.748579,0.87315,0.536524,0.854237,0.660538,0.657183,0.770947,0.808677,0.935335,0.679673,0.768186,0.904326,0.504205,0.679549,0.687832,0.884477,0.908894,0.782707,0.96377,0.576802,0.814802,0.87629,0.808597,0.586707,0.785327,0.992473,0.501568,0.853005,0.961,0.530132,0.934764,0.571185,0.904918,0.96103,0.526692,0.612808,0.962062,0.879838,0.943942,0.830187,0.952573,0.891753,0.6989,0.904111,0.886497,0.863131,0.650068,0.695596,0.884469,0.774664,0.778103,0.576889,0.76839,0.823913,0.507285,0.946384,0.877948,0.672671,0.574615,0.557437,0.842808,0.574349,0.583865,0.514261,0.683169,0.521832,0.929438,0.564866,0.701592,0.660114,0.541049,0.874846};
    vector<double> nine = {0.90554,0.913866,0.844234,0.536289,0.909058,0.539988,0.581528,0.743246,0.737504,0.737017,0.553175,0.710789,0.7377,0.531676,0.884312,0.627669,0.725086,0.522843,0.914021,0.952712,0.726081,0.751511,0.647732,0.938771,0.916815,0.902299,0.934529,0.629814,0.777503,0.987364,0.63228,0.732896,0.787389,0.640546,0.65423,0.641263,0.708138,0.670276,0.834802,0.50936,0.819778,0.503531,0.838154,0.859051,0.566346,0.57782,0.927025,0.513923,0.500286,0.803105,0.792886,0.531074,0.755162,0.515738,0.504323,0.657328,0.70766,0.638633,0.507782,0.792972,0.982869,0.580632,0.680808,0.83312,0.746251,0.746203,0.938463,0.752193,0.615903,0.982413,0.907757,0.677534,0.820306,0.89004,0.906235,0.589177,0.798515,0.633294,0.777335,0.676238,0.534539,0.501836,0.851779,0.841225,0.97045,0.850221,0.666512,0.571801,0.764934,0.740404,0.971052,0.969239,0.502776,0.661475,0.907853,0.779577,0.842464,0.788283,0.670939,0.977141,0.806037,0.572144,0.527149,0.79192,0.800953,0.610094,0.844308,0.785644,0.812331,0.839991,0.736688,0.508423,0.568255,0.657282,0.943387,0.513239,0.503451,0.502314,0.89745,0.94973,0.609378,0.817491,0.574684,0.71349,0.618339,0.925504,0.941476,0.894817,0.69198,0.599705,0.733928,0.635213,0.525984,0.710736,0.831908,0.874228,0.646742,0.796,0.876097,0.562454,0.658319,0.86584,0.677256,0.634507,0.651292,0.765533,0.811517,0.664515,0.998736,0.748238,0.632387,0.530085,0.638524,0.679695,0.630649,0.813908,0.853492,0.643109,0.736898,0.541274,0.686205,0.54434,0.718213,0.508892,0.946517,0.60673,0.806671,0.724862,0.754031,0.997527,0.942993,0.879103,0.590034,0.694679,0.962699,0.575812,0.680179,0.76641,0.553892,0.765284,0.632048,0.828739,0.622156,0.570187,0.641195,0.569565,0.684395,0.628134,0.546818,0.874846,0.540063,0.834432,0.798905,0.704521,0.890012,0.928294,0.839531,0.992699,0.787038,0.747681,0.776647,0.613916,0.591306,0.578867,0.515456,0.765582,0.631212,0.774756,0.822156,0.969805,0.508997,0.720465,0.856404,0.575816,0.739714,0.874364,0.943578,0.716838,0.892423,0.95127,0.990551,0.694089,0.548955,0.778425,0.984699,0.828476,0.700206,0.864891,0.722721,0.763803,0.73227,0.764624,0.532867,0.897195,0.655338,0.772722,0.63236,0.578097,0.568184,0.961127,0.66823,0.940413,0.522563,0.722434,0.941769,0.815244,0.802078,0.532361,0.883894,0.598223,0.83799,0.590383,0.561106,0.505633,0.668381,0.976865,0.670092,0.729106,0.592349,0.6156,0.894255,0.745552,0.989416,0.610039,0.925981,0.970427,0.972838,0.980137,0.660968,0.884897,0.9706,0.868567,0.874846};
    vector<double> ten = {0.501368,0.985561,0.828434,0.983757,0.501151,0.844711,0.554776,0.621056,0.58383,0.927565,0.584419,0.824459,0.680949,0.701681,0.645148,0.507504,0.626413,0.629773,0.595615,0.994497,0.507373,0.922852,0.876295,0.883519,0.797953,0.695076,0.650186,0.682332,0.949296,0.819111,0.802067,0.839257,0.886278,0.670306,0.83396,0.85795,0.571622,0.750295,0.699789,0.856996,0.524924,0.890598,0.772493,0.792133,0.879286,0.6599,0.931695,0.505711,0.984784,0.764368,0.74089,0.639946,0.574449,0.7697,0.845747,0.969794,0.824174,0.889829,0.858917,0.823626,0.675019,0.5437,0.961444,0.987117,0.968151,0.711186,0.89724,0.916754,0.886823,0.840749,0.961456,0.693561,0.671926,0.568375,0.67042,0.750943,0.59224,0.783635,0.554958,0.674154,0.501136,0.586175,0.835483,0.954842,0.535372,0.994644,0.986658,0.752607,0.563891,0.82421,0.500024,0.899877,0.725978,0.52053,0.540515,0.942845,0.891187,0.674099,0.574411,0.630912,0.731479,0.963396,0.800406,0.927912,0.910129,0.54205,0.736828,0.871055,0.827277,0.547552,0.703863,0.819636,0.621579,0.878102,0.757043,0.62802,0.635498,0.808146,0.5085,0.863139,0.781637,0.977353,0.865541,0.642032,0.628131,0.99977,0.635433,0.718668,0.659164,0.568658,0.931373,0.581301,0.933049,0.749187,0.586355,0.864299,0.77616,0.928469,0.783674,0.706687,0.786155,0.900933,0.987935,0.728027,0.952645,0.60522,0.935644,0.875909,0.901477,0.624141,0.94347,0.892773,0.840523,0.663444,0.509579,0.99875,0.997754,0.75438,0.861165,0.592705,0.598374,0.877948,0.66684,0.578963,0.631472,0.64846,0.662757,0.959271,0.966022,0.928955,0.94601,0.583231,0.855818,0.736395,0.590931,0.784826,0.576622,0.787222,0.837943,0.80114,0.764398,0.742521,0.554477,0.597575,0.949906,0.561829,0.658841,0.64885,0.723857,0.862049,0.964791,0.740676,0.536067,0.685303,0.886541,0.591649,0.845077,0.717013,0.831999,0.912959,0.593962,0.724989,0.890559,0.62985,0.895504,0.737267,0.750942,0.581913,0.719524,0.533735,0.985305,0.514766,0.669528,0.760111,0.692848,0.703764,0.667368,0.955666,0.882704,0.605117,0.70631,0.957396,0.960471,0.644366,0.862773,0.623564,0.744512,0.518443,0.977505,0.919731,0.919957,0.714267,0.680874,0.952733,0.581841,0.998978,0.821061,0.568492,0.650742,0.520552,0.918126,0.943816,0.714575,0.854989,0.794466,0.583213,0.563644,0.667688,0.833043,0.958101,0.810115,0.602878,0.56833,0.92353,0.770976,0.78733,0.65492,0.745573,0.851604,0.915799,0.829082,0.881233,0.87505,0.973362,0.791717,0.886265,0.949669,0.586405,0.71051,0.536428,0.742721,0.911888,0.874846};
    
    
    weights_standard.push_back(one);
    weights_standard.push_back(two);
    weights_standard.push_back(three);
    weights_standard.push_back(four);
    weights_standard.push_back(five);
    weights_standard.push_back(six);
    weights_standard.push_back(seven);
    weights_standard.push_back(eight);
    weights_standard.push_back(nine);
    weights_standard.push_back(ten);
    
    
    if (run_simulation) {
        
        //File deleting
        remove("Location");
        remove("values");
        remove("coordinate");
        
        //First set up environment
        int number_of_rovers = 10;
        int number_of_obstacles = 5;
        double radius_of_obstacles = 3;
        double distance_between_rovers = 1;
        double safe_distance = 0.25;
        
        //vectors of rovers
        vector<Rover> teamRover;
        vector<Rover>* p_rover = &teamRover;
        Rover a;
        for (int i=0; i<number_of_rovers; i++) {
            teamRover.push_back(a);
        }
        
        for (int i=0 ; i<number_of_rovers; i++) {
            teamRover.at(i).x_position_vec.push_back(0+(distance_between_rovers*i));
            teamRover.at(i).x_position = teamRover.at(i).x_position_vec.at(0);
            teamRover.at(i).y_position_vec.push_back(0);
            teamRover.at(i).y_position = teamRover.at(i).y_position_vec.at(0);
        }
        
        assert(teamRover.size() == number_of_rovers);
        
        
        //Second set up neural networks
        //Create numNN of neural network with pointer
        int numNN = 1;
        vector<unsigned> topology;
        topology.clear();
        topology.push_back(12);
        topology.push_back(20);
        topology.push_back(2);
        
        for (int rover_number =0 ; rover_number < number_of_rovers; rover_number++) {
            teamRover.at(rover_number).create_neural_network_population(numNN, topology);
        }
        
        POI individualPOI;
        POI* p_poi = &individualPOI;
        
        double rand_1 = random_global(1000);
        double rand_2 = random_global(1000);
        
        individualPOI.x_position_poi_vec.push_back(rand_1);
        individualPOI.y_position_poi_vec.push_back(rand_2);
        individualPOI.value_poi_vec.push_back(100.0);
        
        double scaling_number = find_scaling_number(p_rover,p_poi);
        
        //Obstacles
        vector<vector<double>> location_obstacle;
        vector<vector<double>>* p_location_obstacle = &location_obstacle;
        
        for (int ob = 0 ; ob < number_of_obstacles; ob++) {
            double rand_1 = random_global(750);
            double rand_2 = random_global(750);

            if (ob == 0 ) {
                rand_1 = 4;
                rand_2 = 4;
            }else if(ob == 1){
                rand_1 = 20;
                rand_2 = 80;
            }else if (ob == 2){
                rand_1 = 70;
                rand_2 = 10;
            }else if (ob == 3){
                rand_1=50;
                rand_2 = 30;
            }else if (ob == 4){
                rand_1 = 30;
                rand_2 = 20;
            }
            vector<double> temp;
            temp.push_back(rand_1);
            temp.push_back(rand_2);

            //Check if rand_1 or rand_2 is on or near

            location_obstacle.push_back(temp);
        }
        
        assert(p_rover->size() == number_of_rovers);
        assert(location_obstacle.size() == number_of_obstacles);
        
        //Destination location
        for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
            teamRover.at(rover_number).destination_x_position = (80+(distance_between_rovers*rover_number));
            teamRover.at(rover_number).destination_y_position = 80;
        }
        
        FILE* p_location;
        p_location = fopen("Location", "a");
        for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
            fprintf(p_location, "%f \t %f \n", p_rover->at(rover_number).x_position,p_rover->at(rover_number).y_position);
        }
        fprintf(p_location, "\n");
        
        for (int poi = 0 ; poi < location_obstacle.size(); poi++) {
            fprintf(p_location, "%f \t %f \n", location_obstacle.at(poi).at(0),location_obstacle.at(poi).at(1));
        }
        fprintf(p_location, "\n");
        for (int de = 0 ; de < p_rover->size(); de++) {
            fprintf(p_location, "%f \t %f \n", p_rover->at(de).destination_x_position,p_rover->at(de).destination_y_position);
        }
        fprintf(p_location, "\n");
        fclose(p_location);
        
        
        for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
            for (int layer = 0 ; layer < teamRover.at(rover_number).network_for_agent.at(0).z_layer.size(); layer++) {
                for (int i = 0,co=0; i< teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).size(); i++ ,co++) {
                    for (int j =0 ; j < teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.size(); j++,co++) {
                        //vector<connect> temp;
                        //cout<<teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.at(j).weight<<"\t";
                        teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.at(j).weight = weights_standard.at(rover_number).at(co);
                        //cout<<j<<"\t";
                    }
                    //cout<<"new"<<endl;
                    //cout<<i<<endl;
                }
                //cout<<layer<<endl;
            }
            //cout<<endl;
        }
        
//        for (int rover_number = 0 ; rover_number < teamRover.size(); rover_number++) {
//            for (int layer = 0 ; layer < teamRover.at(rover_number).network_for_agent.at(0).z_layer.size(); layer++) {
//                for (int i = 0,co=0; i< teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).size(); i++ ,co++) {
//                    for (int j =0 ; j < teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.size(); j++,co++) {
//                        //vector<connect> temp;
//                        cout<<teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.at(j).weight<<"\t";
//                        //teamRover.at(rover_number).network_for_agent.at(0).z_layer.at(layer).at(i).z_outputWeights.at(j).weight = weights_standard.at(rover_number).at(co);
//                        //cout<<j<<"\t";
//                    }
//                    //cout<<"new"<<endl;
//                    cout<<i<<endl;
//                }
//                cout<<layer<<endl;
//            }
//            cout<<endl;
//        }
        
        
        
        //Generations
        for(int generation =0 ; generation < 10000 ;generation++){
            cout<<generation<<"Generation"<<endl;
//            cout<<"Size"<<teamRover.size()<<endl;

            simulation(p_rover, p_poi, scaling_number, p_location_obstacle);
            cal_new_way(p_rover,p_location_obstacle,distance_between_rovers,radius_of_obstacles,safe_distance);
            
            if (generation == 0) {
                FILE* p_file;
                p_file = fopen("coordinate_0", "a");
                for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
                    for (int x = 0; x < p_rover->at(rover_number).network_for_agent.at(0).store_x_values.size(); x++) {
                        fprintf(p_file, "%f \t %f \n",p_rover->at(rover_number).network_for_agent.at(0).store_x_values.at(x),p_rover->at(rover_number).network_for_agent.at(0).store_y_values.at(x));
                    }
                    fprintf(p_file, "\n");
                }
                fclose(p_file);
            
                FILE* p_f;
                p_f = fopen("values", "a");
                for (int rover_number = 0 ; rover_number< p_rover->size(); rover_number++) {
                    fprintf(p_f, "%f \t %f \t %f \n", p_rover->at(rover_number).network_for_agent.at(0).collision_with_agents, p_rover->at(rover_number).network_for_agent.at(0).collision_with_obstacles,p_rover->at(rover_number).network_for_agent.at(0).path_value);
                }
                fclose(p_f);
            }
            
//            FILE* p_all;
//            p_all = fopen("coordinate", "a");
//            for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
//                for (int x = 0; x < p_rover->at(rover_number).network_for_agent.at(0).store_x_values.size(); x++) {
//                    fprintf(p_all, "%f \t %f \n",p_rover->at(rover_number).network_for_agent.at(0).store_x_values.at(x),p_rover->at(rover_number).network_for_agent.at(0).store_y_values.at(x));
//                }
////                fprintf(p_all, "\n");
//            }
//            fprintf(p_all, "\n\n");
//            fclose(p_all);
//
//            FILE* p_values_all;
//            p_values_all = fopen("values_all", "a");
//            for (int rover_number = 0 ; rover_number< p_rover->size(); rover_number++) {
//                fprintf(p_values_all, "%f \t %f \t %f \n", p_rover->at(rover_number).network_for_agent.at(0).collision_with_agents, p_rover->at(rover_number).network_for_agent.at(0).collision_with_obstacles,p_rover->at(rover_number).network_for_agent.at(0).path_value);
//            }
//            fprintf(p_values_all, "\n\n");
//            fclose(p_values_all);
            if (generation == 4999) {
                FILE* p_file;
                p_file = fopen("coordinate_4999", "a");
                for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
                    for (int x = 0; x < p_rover->at(rover_number).network_for_agent.at(0).store_x_values.size(); x++) {
                        fprintf(p_file, "%f \t %f \n",p_rover->at(rover_number).network_for_agent.at(0).store_x_values.at(x),p_rover->at(rover_number).network_for_agent.at(0).store_y_values.at(x));
                    }
                    fprintf(p_file, "\n");
                }
                fclose(p_file);
                
                
                FILE* p_f;
                p_f = fopen("values_4999", "a");
                for (int rover_number = 0 ; rover_number< p_rover->size(); rover_number++) {
                    fprintf(p_f, "%f \t %f \t %f \n", p_rover->at(rover_number).network_for_agent.at(0).collision_with_agents, p_rover->at(rover_number).network_for_agent.at(0).collision_with_obstacles,p_rover->at(rover_number).network_for_agent.at(0).path_value);
                }
                fclose(p_f);
                
            }
            
            
            if (generation == 9999) {
                FILE* p_file;
                p_file = fopen("coordinate_9999", "a");
                for (int rover_number = 0 ; rover_number < p_rover->size(); rover_number++) {
                    for (int x = 0; x < p_rover->at(rover_number).network_for_agent.at(0).store_x_values.size(); x++) {
                        fprintf(p_file, "%f \t %f \n",p_rover->at(rover_number).network_for_agent.at(0).store_x_values.at(x),p_rover->at(rover_number).network_for_agent.at(0).store_y_values.at(x));
                    }
                    fprintf(p_file, "\n");
                }
                fclose(p_file);
                
                
                FILE* p_f;
                p_f = fopen("values_9999", "a");
                for (int rover_number = 0 ; rover_number< p_rover->size(); rover_number++) {
                    fprintf(p_f, "%f \t %f \t %f \n", p_rover->at(rover_number).network_for_agent.at(0).collision_with_agents, p_rover->at(rover_number).network_for_agent.at(0).collision_with_obstacles,p_rover->at(rover_number).network_for_agent.at(0).path_value);
                }
                fclose(p_f);
                
            }
            EA_working(p_rover,number_of_rovers);
//            nsgaii(p_rover, number_of_rovers);
//            sort_remove(p_rover, number_of_rovers);
            clean(p_rover);
            
            //Set back
            for (int i=0 ; i<number_of_rovers; i++) {
                teamRover.at(i).x_position_vec.push_back(0+(distance_between_rovers*i));
                teamRover.at(i).x_position = teamRover.at(i).x_position_vec.at(0);
                teamRover.at(i).y_position_vec.push_back(0);
                teamRover.at(i).y_position = teamRover.at(i).y_position_vec.at(0);
            }
            
        }
        
        
    }
    return 0;
}

