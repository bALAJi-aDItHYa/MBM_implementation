`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.12.2020 11:12:41
// Design Name: 
// Module Name: Output_barrel_shifter
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


module Output_barrel_shifter #(parameter N=8,
                               parameter L=3)
                               
                               (input [N-1:0] B1,
                                input [N-1:0] B2,
                                input [L:0] char,
                                input [N:0] mantissa,
                                output [2*N-1:0] product);

wire [2*N-1:0] tmp_8, tmp_4, tmp_2, tmp_1;
wire [2*N-1:0] in, out;

assign in = {{N-2{1'b0}}, mantissa};

assign tmp_8 = char[3]? {in[7:0], in[15:8]}: in;
assign tmp_4 = char[2]? {tmp_8[11:0], tmp_8[15:12]}: tmp_8;
assign tmp_2 = char[1]? {tmp_4[13:0], tmp_4[15:14]}: tmp_4;
assign tmp_1 = char[0]? {tmp_2[14:0], tmp_2[15]}: tmp_2;

//assign product = tmp_1;
/*Corner case (ii) - if the characteristic value is too small (<6)*/
assign out = (char <= 6)? {{N-1{1'b0}}, tmp_1[2*N-1:N-1]}: {tmp_1[N-2:0], tmp_1[2*N-1:N-1]};
assign product = (B1 !=0 && B2 !=0)? out: {2*N-1{1'b0}};
endmodule
