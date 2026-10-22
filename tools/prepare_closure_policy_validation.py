"""Freeze model-selection rules before evaluating new trained policies on the grid."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parent=Path('outputs/support_reference_policy_transfer_v1');grid=Path('outputs/support_reference_validation_v1/protocol.json');training=Path('outputs/closure_margin_comparison_protocol_v1/protocol.json')
    m=json.loads((parent/'manifest.json').read_text());a=json.loads((parent/'independent_training_audit.json').read_text());assert a['passed'] and a['checkpoint_sha256']==sha(parent/'policy.zip')
    successes=[r['case'] for r in a['trajectories'] if r['label']=='after' and r['independently_accepted']]
    output=Path('outputs/closure_policy_validation_protocol_v1');output.mkdir(exist_ok=False)
    protocol=dict(parent=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),parent_known_successes=successes,
        grid_protocol_sha256=sha(grid),training_protocol_sha256=sha(training),
        candidates=['outputs/closure_margin_control98304_v1','outputs/closure_margin_penalty98304_v1'],
        parent_grid_evaluation='outputs/support_parent_policy_validation_v1',
        eligibility='Only prescribed final98304-additional-step checkpoints with complete independent five-case audit and no loss of a parent-known success. Reject ineligible candidates before grid evaluation.',
        selection='Evaluate each eligible candidate on all12 frozen grid cases. Candidate must retain every parent success in both known cases and grid, and add at least one success across their union. If both qualify prefer greater total successes, then the zero-cost control on an exact tie. Otherwise retain parent.',
        scope='Prospective second-stage model selection after reference-grid diagnostics, before new trained-policy grid outcomes. The grid becomes a validation/selection set, not an independent test of a selected model. Single seed, no statistical or hardware claim.',source_sha256=sha(__file__))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'protocol_source.py').write_bytes(Path(__file__).read_bytes());print(output/'protocol.json')


if __name__=='__main__':main()
