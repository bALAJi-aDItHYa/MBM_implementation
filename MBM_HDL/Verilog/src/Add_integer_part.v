`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.12.2020 10:13:31
// Design Name: 
// Module Name: Add_integer_part
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


module Add_integer_part #(parameter N=8,
                          parameter L=3)
                          
                          (input [L-1:0] k1,
                           input [L-1:0] k2,
                           input carry,
                           output [L:0] characteristic);
                           
assign characteristic = k1+k2+carry;

endmodule
