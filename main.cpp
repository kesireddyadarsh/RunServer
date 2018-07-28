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
        //z_outputWeights.back().weight = randomWeight() - 0.5;
        //(((float) rand() / RAND_MAX) * diff) + smallNumber;
        z_outputWeights.back().weight = ((((double) rand() / RAND_MAX) * 1.0) - 0.5);
//        cout<<z_outputWeights.back().weight<<endl;
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
    
    for (int time_step = 1; time_step <120; time_step++) {
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


void  cal(vector<Rover>* teamRover, vector<vector<double>>* p_location_obstacle, double distance_between_rovers,double radius_of_obstacles, double safe_distance){
    
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
        
//
//        for (int index = 0 ; index < teamRover->at(rover_number).network_for_agent.at(0).store_x_values.size(); index++) {
//
//            //Calculate distance to destination
//            double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index),teamRover->at(rover_number).network_for_agent.at(0).store_y_values.at(index) , teamRover->at(rover_number).destination_x_position, teamRover->at(rover_number).destination_y_position);
////            teamRover->at(rover_number).network_for_agent.at(0).path_value +=distance;
//
//            //find if its in closed area or open area
//            for (int ob = 0 ; ob < p_location_obstacle->size(); ob++) {
//                double distance = cal_distance(teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), teamRover->at(rover_number).network_for_agent.at(0).store_x_values.at(index), p_location_obstacle->at(ob).at(0), p_location_obstacle->at(ob).at(0));
//                if (distance < 5) {
//                    open_area = false;
//                    break;
//                }
//            }
//
//            //distance to destination
//
//            if (open_area) {
//                //Its in open area
//                teamRover->at(rover_number).network_for_agent.at(0).path_value += 10;
//            }else{
//                //Its in Closed area
//                teamRover->at(rover_number).network_for_agent.at(0).path_value += 1;
//            }
//
//        }
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
    }else{
        //otherrover is better
        second_policy.at(0) =true;
    }
    
    //Second check for collision obstacle
    if (teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles <= teamRover->at(other_rover).network_for_agent.at(0).collision_with_obstacles) {
        //rover_number is better
        first_policy.at(0) =true;
    }else{
        //otherrover is better
        second_policy.at(0) =true;
    }
    
    //Third check for best path
    if (teamRover->at(rover_number).network_for_agent.at(0).path_value <= teamRover->at(other_rover).network_for_agent.at(0).path_value) {
        //rover_number is better
        first_policy.at(0) =true;
    }else{
        //otherrover is better
        second_policy.at(0) =true;
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

    
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        //No Agents
        teamRover->at(rover_number).network_for_agent.at(0).summation = teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents+teamRover->at(rover_number).network_for_agent.at(0).path_value;
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
        teamRover->at(rover_number).network_for_agent.at(0).path_value = 0;
    }
}

void new_sort(vector<Rover>* teamRover, int number_of_rovers,int number_of_objectives){
    
   double x[1001], y[1001], z[1001];
    for (int rover_number = 0 ; rover_number < teamRover->size(); rover_number++) {
        x[rover_number] = (teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents);
        y[rover_number] = (teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles);
        z[rover_number] = (teamRover->at(rover_number).network_for_agent.at(0).path_value);
    }
    
    for(int i=0; i<number_of_rovers; i++)
    {
        for(int j=0; j<number_of_rovers; j++)
        {
            if(x[j]>=x[j+1])
            {
                double tx, ty, tz;
                tx=x[j];
                x[j]=x[j+1];
                x[j+1]=tx;
                ty=y[j];
                y[j]=y[j+1];
                y[j+1]=ty;
                tz=z[j];
                z[j]=z[j+1];
                z[j+1]=tz;
            }
            if(x[j]==x[j+1])
            {
                if(y[j]>=y[j+1])
                {
                    double ty, tz;
                    ty=y[j];
                    y[j]=y[j+1];
                    y[j+1]=ty;
                    tz=z[j];
                    z[j]=z[j+1];
                    z[j+1]=tz;
                }
            }
            if(x[j]==x[j+1] && y[j]==y[j+1])
            {
                if(z[j]>=z[j+1])
                {
                    double tz;
                    tz=z[j];
                    z[j]=z[j+1];
                    z[j+1]=tz;
                }
            }
        }
    }
    
    //showing results
    for(int counter=1; counter<=number_of_rovers/2; ++counter)
    {
//        cout<<x[counter]<<" "<<y[counter]<<" "<<z[counter]<<endl;
        for (int rover_number = 0 ;rover_number < teamRover->size(); rover_number++) {
            if (x[counter] == teamRover->at(rover_number).network_for_agent.at(0).collision_with_agents) {
                if (y[counter] == teamRover->at(rover_number).network_for_agent.at(0).collision_with_obstacles) {
                    if (z[counter] == teamRover->at(rover_number).network_for_agent.at(0).path_value) {
                        teamRover->erase(teamRover->begin()+rover_number);
                    }
                }
            }
        }
    }
    
    assert(teamRover->size() == (number_of_rovers/2));
    

    
    for (int index = teamRover->size(); index < number_of_rovers; index++) {
        int num = rand()%teamRover->size();
        teamRover->push_back(teamRover->at(num));
        teamRover->at(num).network_for_agent.at(0).mutate();
    }
    
    assert(teamRover->size() == number_of_rovers);
    
    
}



/***************************
 Main
 **************************/

int main(int argc, const char * argv[]) {
    srand((unsigned)time(NULL));
    
    if (run_simulation) {
        
        //File deleting
        remove("Location");
        remove("values");
        remove("coordinate");
        remove("coordinate_0");
        remove("coordinate_4999");
        remove("coordinate_9999");
        remove("values_4999");
        remove("values_9999");
        
        //First set up environment
        int number_of_rovers = 10;
        int number_of_obstacles = 5;
        double radius_of_obstacles = 0.5;
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
                rand_1 = 9;
                rand_2 = 9;
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
            teamRover.at(rover_number).destination_x_position = (100+(distance_between_rovers*rover_number));
            teamRover.at(rover_number).destination_y_position = 100;
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
        

        
        //Generations
        for(int generation =0 ; generation < 200; generation++){
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
            
            if (generation == 99) {
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
            
            
            if (generation == 199) {
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
////            sort_remove(p_rover, number_of_rovers);
//            new_sort(p_rover, number_of_rovers, 3);
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

