`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.12.2020 10:09:56
// Design Name: 
// Module Name: Add_fractional_part
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


module Add_fractional_part #(parameter N=8,
                             parameter L=3)
                             
                             (input [N-2:0] x1,
                              input [N-2:0] x2,
                              output carry,
                              output [N-2:0] fractional);
                              
assign {carry, fractional} = x1+x2;

endmodule
