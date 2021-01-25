`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.12.2020 10:20:06
// Design Name: 
// Module Name: BarrelShift
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

//`include "required_params.v"

module BarrelShift # (parameter N=8,
                      parameter L=3)
                     
                     (input [N-1:0] B,
                      input [L-1:0] k,
                      output [N-2:0] x);
                      
wire [N-1:0] tmp_4, tmp_2, tmp_1;

assign tmp_4 = (k[2])? {B[3:0], B[7:4]}: B;
assign tmp_2 = (k[1])? {tmp_4[1:0], tmp_4[7:2]}: tmp_4;
assign tmp_1 = (k[0])? {tmp_2[0], tmp_2[7:1]}: tmp_2;

//wire [L-1:0] shift;
//assign shift = N-k-1; 

assign x = tmp_1[7:1];
endmodule
