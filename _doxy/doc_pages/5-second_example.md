@page second Second Example
@tableofcontents

For the second example in DASH-Sim, a workload composed of WiFi-TX and WiFi-RX applications (see @ref application) is executed in DASH-Sim. \n
First of all, a heterogeneous SoC with a total of 16 general-purpose cores and hardware accelerators is chosen for this example: 
    - 4 big Arm Cortex-A15 cores,
    - 4 LITTLE Arm Cortex-A7 cores, 
    - 2 matrix multiplier accelerators, 
    - 4 FFT accelerator, and 
    - 2 Viterbi decoder accelerators.
    
DASH-Sim application benchmark suite has 6 reference design of real-world applications from wireless communications and radar domains on \n
two popular commercial heterogeneous SoC platforms: <a href=https://www.xilinx.com/products/boards-and-kits/ek-u1-zcu102-g.html#documentation>Xilinx Zynq ZCU102 UltraScale MpSoC</a> and 
<a href=https://wiki.odroid.com/old_product/odroid-xu3/odroid-xu3>Odroid-XU3</a>. These applications are:
    - WiFi-TX,
    - WiFi-RX,
    - Single-carrier transmitter (SCT)
    - Single-carrier receiver (SCR)
    - Temporal mitigation, and
    - Range detection

The table below gives the execution time profiling for WiFi-TX and WiFi-RX applications.   
<table>
<caption id="execution time profile">Execution time profiles of applications on Arm A7/A15 cores in Odroid-XU3, and hardware accelerators</caption>
<tr><th>Application                   <th>Task                  <th> Odroid A7  <th> Odroid A15     <th>FFT     <th> MM     <th> Viterbi
<tr><td rowspan="6"> WiFi-TX          <td>Scrambler-Endoceder   <td> 22 us      <td> 10 us          <td>-       <td> -      <td> -
<tr>                                  <td>Interleaver           <td> 10 us      <td> 4 us           <td>-       <td> -      <td> -                                                           
<tr>                                  <td>QPSK Modulation       <td> 15 us      <td> 8 us           <td>-       <td> -      <td> -  
<tr>                                  <td>Pilot Insertion       <td> 5 us       <td> 3 us           <td>-       <td> -      <td> -  
<tr>                                  <td>Inverse-FFT           <td> 296 us     <td> 118 us         <td>16 us   <td> -      <td> -  
<tr>                                  <td>CRC                   <td> 5 us       <td> 3 us           <td>-       <td> -      <td> - 
<tr><td rowspan="8"> WiFi-RX          <td>Match Filter          <td> 16 us      <td> 5 us           <td>-       <td> -      <td> -
<tr>                                  <td>Payload Extraction    <td> 8 us       <td> 4 us           <td>-       <td> -      <td> -                                                           
<tr>                                  <td>FFT                   <td> 290 us     <td> 115 us         <td>12 us   <td> -      <td> -  
<tr>                                  <td>Pilot Extraction      <td> 5 us       <td> 3 us           <td>-       <td> -      <td> -  
<tr>                                  <td>QPSK Demodulation     <td> 191 us     <td> 95 us          <td>-       <td> -      <td> -  
<tr>                                  <td>Deinterleaver         <td> 16 us      <td> 9 us           <td>-       <td> -      <td> - 
<tr>                                  <td>Decoder               <td> 1828 us    <td> 738 us         <td>-       <td> -      <td> 2 us
<tr>                                  <td>Descrambler           <td> 3 us       <td> 2 us           <td>-       <td> -      <td> - 
</table>


For this specific workload, We use the parameters (\f$p_{RX} = 0.2, p_{TX} = 0.8\f$) representing the probabilities for the new job being WiFi-RX, WiFi-TX. \n
In other words, for every 4 WiFi-TX applications and 1 WiFi-RX will be present in the workload.

Finally, when we executed the workload, DASH-Sim outputs results in the following format (see @ref second_config)

@image html Streaming_results.png DASH-Sim results for executing a WiFi TX/RX workload. height=40% width=40% 

The above results is only for one frame rate under one scheduler. If DASH-Sim is run for different frame rates (sweeping frame rates) under different scheduler,\n
The following results can be obtained for average execution time.

@image html WiFi_Exec.png Results from different schedulers with a workload consisting of WiFi TX/RX height=30% width=30% 
 
@section second_config Configuration file to run second example

Configuration file: config_file_second_example.ini
