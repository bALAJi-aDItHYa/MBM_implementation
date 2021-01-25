`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.12.2020 09:24:48
// Design Name: 
// Module Name: LOD
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

module LOD #(parameter N= 8,
             parameter L= 3)
             
             (input [N-1:0] B,
              output [L-1:0] k);

assign k = (B[7] == 1'b1)? 3'b111 :
           B[6] == 1'b1  ? 3'b110 :
           B[5] == 1'b1  ? 3'b101 :
           B[4] == 1'b1  ? 3'b100 :
           B[3] == 1'b1  ? 3'b011 :
           B[2] == 1'b1  ? 3'b010 : 
           B[1] == 1'b1  ? 3'b001 :
           3'b000;

endmodule
