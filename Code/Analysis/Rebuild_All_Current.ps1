param(
    [string]$PythonExe = 'python',
    [switch]$SingleRepresentative
)

$ErrorActionPreference = 'Stop'
$Script:RebuildFailed = $false
$Package = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Parent = Split-Path $Package -Parent
$Source = Join-Path $Package 'Source_Data'
$Core = Join-Path $Package 'Code\Core'
$Main = Join-Path $Core 'PC_AEGG_主程序.py'
$Helper = Join-Path $PSScriptRoot 'rebuild_ablation_outputs.py'
$Python = (Get-Command $PythonExe -ErrorAction Stop).Source
$WaterArguments = @('--gamma','0.072','--contact-angle','15','--rho','998','--surface-gas-density','0','--gravity','9.80665','--viscosity','0.001','--reference-flow-rate','2e-7','--surface-diam','0.30','--internal-diam-bottom','0.50','--internal-diam-top','0.40','--slice-dz','1','--capillary-band-start-mm','5','--capillary-min-ratio','1.15','--capillary-assembly-reserve-ratio','0.02')
$StageRoot = Join-Path $Parent '_rebuild_stage_20261010'
$BackupRoot = Join-Path $Parent '_rebuild_previous_20261010'
$StatusFile = Join-Path $StageRoot 'REBUILD_STATUS.csv'
$Cuboid = Join-Path $Package 'Input_Geometries\长方体.stl'
New-Item -ItemType Directory -Path $StageRoot,$BackupRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }

function Sha($Path) {
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'SHA256 input path is empty' }
    $stream = [IO.File]::OpenRead([IO.Path]::GetFullPath($Path))
    try {
        $hash = [Security.Cryptography.SHA256]::Create().ComputeHash($stream)
        return [BitConverter]::ToString($hash).Replace('-', '').ToLowerInvariant()
    } finally { $stream.Dispose() }
}
$CoreHashes = @{}
Get-ChildItem -LiteralPath $Core -Filter '*.py' -File | ForEach-Object { $CoreHashes[$_.Name] = Sha $_.FullName }
$HelperHash = Sha $Helper

function Check-Code {
    foreach ($name in $CoreHashes.Keys) {
        if ((Sha (Join-Path $Core $name)) -ne $CoreHashes[$name]) {
            throw "Core code changed during rebuild: $name"
        }
    }
    if ((Sha $Helper) -ne $HelperHash) { throw 'Ablation helper changed during rebuild' }
}

function Record($Id, $Status, $Detail, $Destination) {
    $row = [pscustomobject]@{
        time = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        item = $Id
        status = $Status
        detail = $Detail
        destination = $Destination
        main_code_sha256 = $CoreHashes['PC_AEGG_主程序.py']
        surface_code_sha256 = $CoreHashes['surface_patch_builder.py']
    }
    $row | Export-Csv -LiteralPath $StatusFile -NoTypeInformation -Encoding UTF8 -Append
    Write-Host "$($row.time) [$Status] $Id $Detail"
}

function Require-File($Dir, $Name) {
    $p = Join-Path $Dir $Name
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing $p" }
    if ((Get-Item -LiteralPath $p).Length -eq 0) { throw "Empty $p" }
}

function Require-Figures($Dir, $Minimum, $Subdir = '') {
    $p = if ($Subdir) { Join-Path $Dir $Subdir } else { $Dir }
    $count = @(Get-ChildItem -LiteralPath $p -Filter '*.svg' -File -ErrorAction Stop).Count
    if ($count -lt $Minimum) { throw "Only $count SVG figures in $p; expected at least $Minimum" }
}

function Validate-Full($Dir) {
    foreach ($name in @('run_config.json','final_stitched.stl','surface_only.stl',
        'internal_only.stl','assembly_nodes_final.csv','assembly_edges_final.csv',
        'post_assembly_pressure_profile.csv','final_assembly_status.json',
        'evolution_internal.csv','evolution_process_forces.csv',
        'plot_forces.svg','plot_porosity_profile.svg','plot_process_evolution.svg')) {
        Require-File $Dir $name
    }
    foreach ($name in @('surface_validation\surface_nodes.csv',
        'surface_validation\surface_edges.csv',
        'surface_validation\surface_refinement_parameter_evolution.csv',
        'global_surface_pore_audit\global_surface_pore_summary.json')) {
        Require-File $Dir $name
    }
    Require-Figures $Dir 7
    Require-Figures $Dir 8 'surface_validation'
    $s = Get-Content (Join-Path $Dir 'final_assembly_status.json') -Raw | ConvertFrom-Json
    foreach ($key in @('graph_screen_pass','dual_graph_screen_pass','internal_print_support_screen_pass',
        'surface_global_complete_pore_acceptance_pass','pressure_profile_pass',
        'fine_slice_pressure_pass','stl_format_pass')) {
        if ($s.$key -ne $true) { throw "Full validation failed: $key" }
    }
}

function Validate-Internal($Dir) {
    foreach ($name in @('internal_only.stl','structure_nodes.csv','structure_edges.csv',
        'internal_slice_physics_analysis.csv','plot_forces.svg',
        'plot_porosity_profile.svg')) { Require-File $Dir $name }
    Require-Figures $Dir 7
}

function Validate-Surface($Dir) {
    Require-File $Dir 'surface_only.stl'
    foreach ($name in @('surface_validation\surface_nodes.csv',
        'surface_validation\surface_edges.csv',
        'surface_validation\surface_refinement_parameter_evolution.csv',
        'global_surface_pore_audit\global_surface_pore_summary.json')) {
        Require-File $Dir $name
    }
    Require-Figures $Dir 8 'surface_validation'
}

function Run-Python($Id, $Arguments, $Log) {
    Check-Code
    $ErrorActionPreference = 'Continue'
    & $Python @Arguments *> $Log
    $rc = $LASTEXITCODE
    if ($rc -ne 0) { throw "Python exit code $rc; see $Log" }
    Check-Code
}

function Stage-Path($Id) {
    $name = '{0}_{1}' -f $Id, (Get-Date -Format 'yyyyMMdd_HHmmss')
    $p = Join-Path $StageRoot $name
    New-Item -ItemType Directory -Path $p -Force | Out-Null
    return $p
}

function Write-Provenance($Dir, $Id, $InputStl, $Kind) {
    $item = [ordered]@{
        item = $Id
        generation_kind = $Kind
        generated_at = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        input_stl = $InputStl
        input_stl_sha256 = Sha $InputStl
        core_code_sha256 = $CoreHashes
        helper_sha256 = $HelperHash
        output_sha256 = [ordered]@{
            "assembly_nodes_final.csv" = if (Test-Path (Join-Path $Dir "assembly_nodes_final.csv")) { Sha (Join-Path $Dir "assembly_nodes_final.csv") } else { $null }
            "assembly_edges_final.csv" = if (Test-Path (Join-Path $Dir "assembly_edges_final.csv")) { Sha (Join-Path $Dir "assembly_edges_final.csv") } else { $null }
            "run_config.json" = if (Test-Path (Join-Path $Dir "run_config.json")) { Sha (Join-Path $Dir "run_config.json") } else { $null }
        }
    }
    $item | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Dir 'generation_provenance.json') -Encoding UTF8
}

function Publish($Id, $Stage, $Destination) {
    $destFull = [IO.Path]::GetFullPath($Destination)
    $rootFull = [IO.Path]::GetFullPath($Package).TrimEnd('\') + '\'
    if (-not $destFull.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Destination outside package: $destFull"
    }
    if (-not (Test-Path -LiteralPath $Stage -PathType Container)) { throw "Missing stage: $Stage" }
    New-Item -ItemType Directory -Path (Split-Path $Destination -Parent) -Force | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        $old = Join-Path $BackupRoot ('{0}_{1}' -f $Id, (Get-Date -Format 'yyyyMMdd_HHmmss'))
        if (Test-Path -LiteralPath $old) { throw "Backup collision: $old" }
        Move-Item -LiteralPath $Destination -Destination $old
    }
    Move-Item -LiteralPath $Stage -Destination $Destination
    Record $Id 'published' 'validated full output' $Destination
}

function Is-Current-Full($Id, $Destination, $InputStl, $Faces, $Density, $Seed) {
    $provenancePath = Join-Path $Destination 'generation_provenance.json'
    if (-not (Test-Path -LiteralPath $provenancePath -PathType Leaf)) { return $false }
    try {
        $p = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
        $cfg = Get-Content -LiteralPath (Join-Path $Destination 'run_config.json') -Raw | ConvertFrom-Json
        if ($p.item -ne $Id -or $p.input_stl_sha256 -ne (Sha $InputStl) -or
            $p.helper_sha256 -ne $HelperHash) { return $false }
        foreach ($name in $CoreHashes.Keys) {
            if ($p.core_code_sha256.$name -ne $CoreHashes[$name]) { return $false }
        }
        if ([int]$cfg.surface.initial_face_count -ne [int]$Faces -or
            [int]$cfg.internal_dynamic.seed -ne [int]$Seed -or
            [math]::Abs([double]$cfg.internal_dynamic.initial_skeleton_density_multiplier_effective - [double]$Density) -gt 1e-12) {
            return $false
        }
        if ($cfg.fluid.surface_tension_N_per_m -ne 0.072 -or
            $cfg.fluid.contact_angle_deg -ne 15 -or $cfg.fluid.density_kg_per_m3 -ne 998 -or
            $cfg.fluid.viscosity_Pa_s -ne 0.001 -or $cfg.surface.diameter_mm -ne 0.30 -or
            $cfg.internal_dynamic.internal_diam_bottom_mm -ne 0.50 -or
            $cfg.internal_dynamic.internal_diam_top_mm -ne 0.40 -or
            $cfg.internal_dynamic.capillary_band_start_mm -ne 5) { return $false }
        foreach ($n in @('assembly_nodes_final.csv','assembly_edges_final.csv','run_config.json')) {
            if ($p.output_sha256.$n -ne (Sha (Join-Path $Destination $n))) { return $false }
        }
        Validate-Full $Destination
        return $true
    } catch { return $false }
}

function Run-Full($Id, $StlName, $Faces, $Density, $Seed, $Destination) {
    $stl = Join-Path $Package "Input_Geometries\$StlName"
    if (Is-Current-Full $Id $Destination $stl $Faces $Density $Seed) {
        Write-Host "Already published with current code and parameters: $Id"; return
    }
    $stage = Stage-Path $Id
    Record $Id 'running' 'full current-code generation' $Destination
    try {
        $args = @($Main,'--stl',$stl,'--height-mm','25','--out-dir',$stage,
            '--build-mode','dynamic','--surface-design','--surface-faces',"$Faces",
            '--initial-density-multiplier',"$Density",'--seed',"$Seed")
        $args += $WaterArguments
        Run-Python $Id $args (Join-Path $StageRoot "$Id.log")
        Validate-Full $stage
        Write-Provenance $stage $Id $stl 'current_code_full_generation'
        Publish $Id $stage $Destination
    } catch {
        Record $Id 'failed' $_.Exception.Message $stage
        $Script:RebuildFailed = $true
        throw
    }
}

$Extended = Join-Path $Source 'Extended_Geometries'
Run-Full 'representative_seed42' '长方体.stl' 1000 0.9 42 (Join-Path $Source 'Representative_Structure_Seed42')
if (-not (Test-Path -LiteralPath (Join-Path $Source 'Representative_Structure_Seed42\generation_provenance.json'))) {
    throw 'Representative seed 42 was not published; stopping before extension and ablation runs.'
}
if ($SingleRepresentative) { return }

$geometry = @(
    @('large_hole_seed42','大孔洞体.stl',2000,1.4,'Large_Hole_seed42'),
    @('gaussian_sign_change_seed42','高斯曲率变号.stl',3000,1.4,'Gaussian_Curvature_Sign_Change_seed42'),
    @('local_high_negative_seed42','局部高负曲率曲面.stl',4000,1.3,'Local_High_Negative_Curvature_seed42'),
    @('smooth_variable_seed42','连续光滑变曲率曲面.stl',3000,1.4,'Smooth_Variable_Curvature_seed42'),
    @('multibranch_seed42','knitted_elephant.stl',6000,1.4,'Multibranch_Free_Boundary_seed42')
)
foreach ($g in $geometry) { Run-Full $g[0] $g[1] $g[2] $g[3] 42 (Join-Path $Extended $g[4]) }

$Ablation = Join-Path $Source 'Ablation_Analysis'
foreach ($seed in @(101,202,303,404,505)) {
    $id = "paired_full_seed$seed"
    $dest = Join-Path $Ablation "Paired_Full_Dual_Domain\$id"
    Run-Full $id '长方体.stl' 1000 0.9 $seed $dest
    if (-not (Test-Path -LiteralPath (Join-Path $dest 'generation_provenance.json'))) { continue }

    $internalId = "internal_full_seed$seed"
    $surfaceId = "surface_full_seed$seed"
    $internalDest = Join-Path $Ablation "内部\Internal_Full_Runs\seed_$seed"
    $surfaceDest = Join-Path $Ablation "表面\Surface_Groups\$surfaceId"
    if ($true) {
        $work = Stage-Path "split_full_seed$seed"
        $intStage = Join-Path $work 'internal'
        $surfStage = Join-Path $work 'surface'
        try {
            $call = @($Helper,'split-full','--source',$dest,
                '--stl',$Cuboid,'--internal-out',$intStage,'--surface-out',$surfStage)
            Run-Python "split_full_seed$seed" $call (Join-Path $StageRoot "split_full_seed$seed.log")
            Validate-Internal $intStage
            Validate-Surface $surfStage
            Write-Provenance $intStage $internalId $Cuboid 'current_code_paired_internal_graph'
            Write-Provenance $surfStage $surfaceId $Cuboid 'current_code_paired_surface_graph'
            Publish $internalId $intStage $internalDest
            Publish $surfaceId $surfStage $surfaceDest
        } catch { $Script:RebuildFailed = $true; Record "split_full_seed$seed" 'failed' $_.Exception.Message $work }
    }

    foreach ($policy in @('random','none')) {
        $controlId = if ($policy -eq 'random') { "surface_random_pore_seed$seed" } else { "surface_none_seed$seed" }
        $controlDest = Join-Path $Ablation "表面\Surface_Groups\$controlId"
        # Recompute derived controls; provenance existence alone is not a cache check.
        $work = Stage-Path $controlId
        try {
            $call = @($Helper,'surface-control','--full',$dest,
                '--stl',$Cuboid,'--out',$work,'--policy',$policy,'--seed',"$seed")
            Run-Python $controlId $call (Join-Path $StageRoot "$controlId.log")
            Validate-Surface $work
            Write-Provenance $work $controlId $Cuboid 'current_code_surface_control_generation'
            Publish $controlId $work $controlDest
        } catch { $Script:RebuildFailed = $true; Record $controlId 'failed' $_.Exception.Message $work }
    }

    $randomId = "internal_random_position_fixed_budget_seed$seed"
    $randomDest = Join-Path $Ablation "内部\Internal_Random_Position_Fixed_Budget\$randomId"
    if ($true) {
        $work = Stage-Path $randomId
        try {
            $intEdges = Import-Csv (Join-Path $dest 'assembly_edges_final.csv') | Where-Object { $_.bond_type -eq 'internal' }
            $endpoints = [Collections.Generic.HashSet[int]]::new()
            foreach ($edge in $intEdges) { [void]$endpoints.Add([int]$edge.source); [void]$endpoints.Add([int]$edge.target) }
            $call = @($Main,'--stl',$Cuboid,'--height-mm','25',
                '--out-dir',$work,'--build-mode','dynamic','--no-surface-design',
                '--internal-only','--internal-matched-random-position',
                '--internal-matched-node-budget',"$($endpoints.Count)",
                '--internal-matched-edge-budget',"$($intEdges.Count)",
                '--initial-density-multiplier','0.9','--seed',"$seed")
            $call += $WaterArguments
            Run-Python $randomId $call (Join-Path $StageRoot "$randomId.log")
            Validate-Internal $work
            Write-Provenance $work $randomId $Cuboid 'current_code_matched_random_position_generation'
            Publish $randomId $work $randomDest
        } catch { $Script:RebuildFailed = $true; Record $randomId 'failed' $_.Exception.Message $work }
    }

    foreach ($control in @(@('initial_gradient','Internal_Initial_Gradient_Skeleton'),
                           @('fully_random','Internal_Fully_Random'))) {
        $kind = $control[0]
        $group = $control[1]
        $idControl = "${kind}_seed$seed"
        $controlDest = Join-Path $Ablation "内部\$group\seed_$seed"
        # Recompute derived controls; provenance existence alone is not a cache check.
        $work = Stage-Path $idControl
        try {
            $call = @($Helper,'retained-control','--source',$controlDest,
                '--stl',$Cuboid,'--out',$work,'--kind',$kind)
            Run-Python $idControl $call (Join-Path $StageRoot "$idControl.log")
            Validate-Internal $work
            Write-Provenance $work $idControl $Cuboid 'retained_control_graph_current_code_analysis_export'
            Publish $idControl $work $controlDest
        } catch { $Script:RebuildFailed = $true; Record $idControl 'failed' $_.Exception.Message $work }
    }
}

if ($Script:RebuildFailed) { throw 'One or more derived runs failed; see REBUILD_STATUS.csv. Final analysis was not published.' }

$pairScript = Join-Path $PSScriptRoot 'pair_ablation_controls.py'
Run-Python 'pair_ablation_controls' @($pairScript) (Join-Path $StageRoot 'pair_ablation_controls.log')

foreach ($script in @('rebuild_final_comparison_tables.py','audit_final_package.py')) {
    $scriptPath = Join-Path $PSScriptRoot $script
    Run-Python $script @($scriptPath) (Join-Path $StageRoot "$script.log")
}
Record 'batch' 'finished' 'Data tables and index rebuilt; figure assembly is separate' $Package
