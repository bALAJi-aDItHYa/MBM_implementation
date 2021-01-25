`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.12.2020 08:40:25
// Design Name: 
// Module Name: Mul_top
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

module Mul_top #(parameter N=8,
                 parameter L=3)
                
                (input [N-1:0] B1,
                 input [N-1:0] B2,
                 output [2*N-1:0] product);

wire [L-1:0] k1,k2;
wire [N-2:0] x1,x2;
wire [N-2:0] fractional;
wire [N:0] mantissa;
wire [L:0] characteristic; 
wire c0, carry;

LOD  #(N,L) one_detector1 (B1,k1);
LOD  #(N,L) one_detector2 (B2,k2);

BarrelShift #(N,L) shifter1 (B1, k1, x1);
BarrelShift #(N,L) shifter2 (B2, k2, x2);

Add_fractional_part #(N,L) Add_fraction (x1, x2, carry, fractional);
Add_integer_part    #(N,L) Add_int(k1, k2, carry, characteristic);
Add_error_coefficient #(N,L) Add_error(fractional, carry, c0, mantissa);
Output_barrel_shifter #(N,L) OP_BarrelShift(B1, B2, characteristic, mantissa, product);

endmodule
