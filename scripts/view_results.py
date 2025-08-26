#!/usr/bin/env python3
"""
Simple command-line script to browse HEP Foundation experiment results.
Useful for NERSC environments where web interfaces are difficult to access.
"""

import os
import sys
from pathlib import Path


def list_experiments():
    """List all available experiments."""
    experiments = {}
    
    # Check test results
    test_path = Path("_test_results/test_foundation_experiments")
    if test_path.exists():
        test_experiments = [d for d in test_path.iterdir() if d.is_dir()]
        if test_experiments:
            experiments["Test Results"] = sorted(test_experiments)
    
    # Check foundation experiments  
    foundation_path = Path("_foundation_experiments")
    if foundation_path.exists():
        foundation_experiments = [d for d in foundation_path.iterdir() if d.is_dir()]
        if foundation_experiments:
            experiments["Foundation Experiments"] = sorted(foundation_experiments)
    
    # Check standalone experiments
    standalone_path = Path("_standalone_experiments")
    if standalone_path.exists():
        standalone_experiments = [d for d in standalone_path.iterdir() if d.is_dir()]
        if standalone_experiments:
            experiments["Standalone Experiments"] = sorted(standalone_experiments)
    
    return experiments


def show_experiment_structure(experiment_path):
    """Show the structure of an experiment directory."""
    exp_path = Path(experiment_path)
    if not exp_path.exists():
        print(f"❌ Experiment not found: {experiment_path}")
        return
    
    print(f"\n📁 Experiment: {exp_path.name}")
    print(f"📍 Path: {exp_path}")
    print("=" * 70)
    
    # Show config file
    config_file = exp_path / "_experiment_config.yaml"
    if config_file.exists():
        print("📄 Configuration file found")
        print(f"   Size: {config_file.stat().st_size} bytes")
    
    # Show directory structure
    subdirs = [d for d in exp_path.iterdir() if d.is_dir()]
    if subdirs:
        print("\n📂 Subdirectories:")
        for subdir in sorted(subdirs):
            print(f"   {subdir.name}/")
            
            # Count files in each subdirectory
            files = list(subdir.glob("*"))
            if files:
                png_files = list(subdir.glob("*.png"))
                other_files = [f for f in files if f.suffix != ".png"]
                
                if png_files:
                    print(f"      🖼️  {len(png_files)} PNG files")
                if other_files:
                    print(f"      📄 {len(other_files)} other files")
    
    # Count all PNG files recursively
    all_pngs = list(exp_path.rglob("*.png"))
    if all_pngs:
        print(f"\n🖼️  Total plots found: {len(all_pngs)}")
        
        # Group by directory
        by_dir = {}
        for png in all_pngs:
            dir_name = png.parent.name
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(png.name)
        
        for dir_name, files in sorted(by_dir.items()):
            print(f"   📁 {dir_name}: {len(files)} plots")
            for file in sorted(files)[:3]:  # Show first 3 files
                print(f"      - {file}")
            if len(files) > 3:
                print(f"      ... and {len(files) - 3} more")


def list_plots(experiment_path):
    """List all plot files in an experiment."""
    exp_path = Path(experiment_path)
    if not exp_path.exists():
        print(f"❌ Experiment not found: {experiment_path}")
        return
    
    png_files = list(exp_path.rglob("*.png"))
    if not png_files:
        print("❌ No plot files found")
        return
    
    print(f"\n🖼️  All plots in {exp_path.name}:")
    print("=" * 70)
    
    # Group by directory
    by_dir = {}
    for png in png_files:
        rel_path = png.relative_to(exp_path)
        dir_name = rel_path.parent
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(png)
    
    for dir_name, files in sorted(by_dir.items()):
        print(f"\n📁 {dir_name}:")
        for file in sorted(files):
            file_size = file.stat().st_size
            print(f"   {file.name} ({file_size:,} bytes)")


def show_config(experiment_path):
    """Show experiment configuration."""
    exp_path = Path(experiment_path)
    config_file = exp_path / "_experiment_config.yaml"
    
    if not config_file.exists():
        print("❌ No configuration file found")
        return
    
    print(f"\n📄 Configuration for {exp_path.name}:")
    print("=" * 70)
    try:
        content = config_file.read_text()
        print(content)
    except Exception as e:
        print(f"❌ Error reading config: {e}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("📊 HEP Foundation Results Viewer")
        print("=" * 40)
        print("Usage:")
        print("  python scripts/view_results.py list                    # List all experiments")
        print("  python scripts/view_results.py show <experiment_name>  # Show experiment structure")
        print("  python scripts/view_results.py plots <experiment_name> # List all plots")
        print("  python scripts/view_results.py config <experiment_name># Show configuration")
        print("\nExamples:")
        print("  python scripts/view_results.py list")
        print("  python scripts/view_results.py show 013_Foundation_VAE_Model")
        print("  python scripts/view_results.py plots 013_Foundation_VAE_Model")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        experiments = list_experiments()
        if not experiments:
            print("❌ No experiments found")
            return
        
        print("📊 Available Experiments:")
        print("=" * 50)
        
        for category, exp_list in experiments.items():
            print(f"\n📂 {category}:")
            for exp in exp_list:
                print(f"   {exp.name}")
        
        print(f"\n💡 To explore an experiment, run:")
        print(f"   python scripts/view_results.py show <experiment_name>")
    
    elif command in ["show", "plots", "config"]:
        if len(sys.argv) < 3:
            print("❌ Please provide experiment name")
            return
        
        exp_name = sys.argv[2]
        
        # Find experiment path
        possible_paths = [
            Path("_test_results/test_foundation_experiments") / exp_name,
            Path("_foundation_experiments") / exp_name,
            Path("_standalone_experiments") / exp_name,
        ]
        
        exp_path = None
        for path in possible_paths:
            if path.exists():
                exp_path = path
                break
        
        if exp_path is None:
            print(f"❌ Experiment '{exp_name}' not found")
            print("💡 Run 'python scripts/view_results.py list' to see available experiments")
            return
        
        if command == "show":
            show_experiment_structure(exp_path)
        elif command == "plots":
            list_plots(exp_path)
        elif command == "config":
            show_config(exp_path)
    
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Use 'list', 'show', 'plots', or 'config'")


if __name__ == "__main__":
    main()