plots:
- Subset of api exploration viz
- data matrix

- The first entry point to pyomeca is object creation
- There are several way
- from scratch, from random_data, from other files or from other data structures

## From scratch

- The first way to create data array in pyomeca is to specify directly the data

!!! example
    === "Angles"
        <div class="template">/api/angles/#pyomeca.angles.Angles</div>
    
    === "Markers"
        <div class="template">/api/markers/#pyomeca.markers.Markers</div>

    === "Rototrans" 
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans</div>

    === "Analogs"
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs</div>

## From random data

!!! Example
    === "Angles"
        <div class="template">/api/angles/#pyomeca.angles.Angles.from_random_data</div>
        
    === "Markers"
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_random_data</div>

    === "Rototrans" 
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_random_data</div>

    === "Analogs"
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_random_data</div>

## From files

=== "c3d"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_c3d</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_c3d</div>

=== "csv"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_csv</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_csv</div>

=== "excel"
    !!! Example
        === "Markers"
            <div class="template">/api/markers/#pyomeca.markers.Markers.from_excel</div>
            
        === "Analogs"
            <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_excel</div>

=== "mot"
    !!! Example
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_mot</div>

=== "trc"
    !!! Example
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_trc</div>

=== "sto"
    !!! Example
        <div class="template">/api/analogs/#pyomeca.analogs.Analogs.from_sto</div>

## From other data structures

### Angles & Rototrans

!!! Example
    === "Angles from Rototrans"
        <div class="template">/api/angles/#pyomeca.angles.Angles.from_rototrans</div>

    === "Rototrans from Angles"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_euler_angles</div>

### Markers & Rototrans

!!! Example
    ===! "Markers from Rototrans"
        <div class="template">/api/markers/#pyomeca.markers.Markers.from_rototrans</div>

    === "Rototrans from Markers"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_markers</div>

### Processed Rototrans

!!! Example
    ===! "Rototrans from a transposed Rototrans"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_transposed_rototrans</div>

    === "Rototrans from an averaged Rototrans"
        <div class="template">/api/rototrans/#pyomeca.rototrans.Rototrans.from_averaged_rototrans</div>
        
<script src="../js/template.js"></script>
<script>
    renderApiTemplate()
</script>
