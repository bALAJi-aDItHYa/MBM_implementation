`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 09.12.2020 10:16:30
// Design Name: 
// Module Name: Add_error_coefficient
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


module Add_error_coefficient #(parameter N=8,
                               parameter L=3)
                               
                               (input [N-2:0] fractional,
                                input carry,
                                output c0,
                                output [N:0] mantissa);

/* MBM Implementation
wire [N-2:0] tmp, tmp_2;
wire [6:0] error_correction;
wire temp_carry;

assign error_correction = 7'b0001010>>carry;
assign {temp_carry, tmp} = fractional + error_correction;

// Corner case (i) - if the val x1 +x2 +error_correction >2
assign {c0,tmp_2} = (carry == 1 && fractional > 7'b1110110)? {1'b0,fractional}: {temp_carry,tmp};

assign mantissa = c0? {2'b10, tmp_2}: {2'b01, tmp_2};*/

//Mitchell Algo implementation
assign c0 = 1'b0;
assign mantissa = {2'b01, fractional};

endmodule
