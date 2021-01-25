`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.12.2020 11:16:31
// Design Name: 
// Module Name: test_bench
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

module test_bench();

parameter N=8;
parameter L=3;

reg [N-1:0] B1;
reg [N-1:0] B2;
wire [2*N-1:0] prod;
integer file1, file2, file_prod;

Mul_top #(N,L) uut (B1, B2, prod);

initial begin
    file1 = $fopen("/home/balaji5199/Desktop/TU_Dresden/B1.txt", "r");
    file2 = $fopen("/home/balaji5199/Desktop/TU_Dresden/B2.txt", "r");
    file_prod = $fopen("/home/balaji5199/Desktop/TU_Dresden/Mitchell_Prod.txt", "w");
    
    while(!$feof(file1)) begin
        $fscanf(file1, "%d\n", B1);
        $fscanf(file2, "%d\n", B2);
        #5;
        $fwrite(file_prod, "%d\n", prod);
    end
    $fclose(file1);
    $fclose(file2);
    $fclose(file_prod);
//B1 = 5;
//B2 = 14;
//#5;

//B1 = 7;
//B2 = 10;
//#5;

//B1 = 10;
//B2 = 7;
//#5;

//B1 = 14;
//B2 = 5;
//#5;
    $finish;
end

endmodule
