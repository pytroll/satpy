#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c14526.ad.smhi.se>
#   Sauli.Joro <sauli.joro@eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''Definition of Header Records for the MSG Level 1.5 data (hrit or native)

NOTE: impf_configuration in L15HeaderRecord-class needs to be fixed!

'''


import numpy as np


class GSDTRecords(object):

    '''MSG Ground Segment Data Type records.
    Reference Document:
            MSG Ground Segment Design Specification (GSDS)
    '''

    @property
    def gp_cpu_address(self):

        record = [
            ('Qualifier_1', np.uint8),
            ('Qualifier_2', np.uint8),
            ('Qualifier_3', np.uint8),
            ('Qualifier_4', np.uint8)
        ]

        return record

    @property
    def gp_fac_env(self):

        return np.uint8

    @property
    def gp_fac_id(self):

        return np.uint8

    @property
    def gp_pk_header(self):

        record = [
            ('HeaderVersionNo', np.uint8),
            ('PacketType', np.uint8),
            ('SubHeaderType', np.uint8),
            ('SourceFacilityId', self.gp_fac_id),
            ('SourceEnvId', self.gp_fac_env),
            ('SourceInstanceId', np.uint8),
            ('SourceSUId', self.gp_su_id),
            ('SourceCPUId', self.gp_cpu_address),
            ('DestFacilityId', self.gp_fac_id),
            ('DestEnvId', self.gp_fac_env),
            ('SequenceCount', np.uint16),
            ('PacketLength', np.int32)
        ]

        return record

    @property
    def gp_pk_sh1(self):

        record = [
            ('SubHeaderVersionNo', np.uint8),
            ('ChecksumFlag', np.bool),
            ('Acknowledgement', (np.uint8, 4)),
            ('ServiceType', self.gp_svce_type),
            ('ServiceSubtype', np.uint8),
            ('PacketTime', self.time_cds_short),
            ('SpacecraftId', self.gp_sc_id)
        ]

        return record

    @property
    def gp_sc_id(self):

        return np.uint16

    @property
    def gp_su_id(self):

        return np.uint32

    @property
    def gp_svce_type(self):

        return np.uint8

    @property
    def time_cds(self):

        record = [
            ('Day', np.uint16),
            ('MilliSecsOfDay', np.uint32),
            ('MicrosecsOfMillisecs', np.uint16)]

        return record

    @property
    def time_cds_expanded(self):

        record = [
            ('Day', np.uint16),
            ('MilliSecsOfDay', np.uint32),
            ('MicrosecsOfMillisecs', np.uint16),
            ('NanosecsOfMicrosecs', np.uint16)
        ]

        return record

    @property
    def time_cds_short(self):

        record = [
            ('Day', np.uint16),
            ('MilliSeconds', np.uint32)
        ]

        return record


class Msg15NativeHeaderRecord(object):

    def get(self):

        record = [
            ('15_MAIN_PRODUCT_HEADER', L15MainProductHeaderRecord().get()),
            ('15_SECONDARY_PRODUCT_HEADER',
             L15SecondaryProductHeaderRecord().get()),
            #('GP_PK_HEADER', GpPkHeaderRecord().get()),
            #('GP_PK_SH1', GpPkSh1Record().get()),
            ('GP_PK_HEADER', GSDTRecords().gp_pk_header),
            ('GP_PK_SH1', GSDTRecords().gp_pk_sh1),
            ('15_DATA_HEADER', L15DataHeaderRecord().get())
        ]

        return record


class L15PhData(object):

    @property
    def l15_ph_data(self):

        record = [
            ('Name', (np.str, 30)),
            ('Value', (np.str, 50))
        ]

        return record


class L15MainProductHeaderRecord(L15PhData):

    '''
    Reference Document:
            MSG Level 1.5 Native Format File Definition
    '''

    def get(self):

        record = [
            ('FormatName', self.l15_ph_data),
            ('FormatDocumentName', self.l15_ph_data),
            ('FormatDocumentMajorVersion', self.l15_ph_data),
            ('FormatDocumentMinorVersion', self.l15_ph_data),
            ('CreationDateTime', self.l15_ph_data),
            ('CreatingCentre', self.l15_ph_data),
            ('DataSetIdentification', (self.l15_ph_data_identification, 27)),
            ('TotalFileSize', self.l15_ph_data),
            ('GORT', self.l15_ph_data),
            ('ASTI', self.l15_ph_data),
            ('LLOS', self.l15_ph_data),
            ('SNIT', self.l15_ph_data),
            ('AIID', self.l15_ph_data),
            ('SSBT', self.l15_ph_data),
            ('SSST', self.l15_ph_data),
            ('RRCC', self.l15_ph_data),
            ('RRBT', self.l15_ph_data),
            ('RRST', self.l15_ph_data),
            ('PPRC', self.l15_ph_data),
            ('PPDT', self.l15_ph_data),
            ('GPLV', self.l15_ph_data),
            ('APNM', self.l15_ph_data),
            ('AARF', self.l15_ph_data),
            ('UUDT', self.l15_ph_data),
            ('QQOV', self.l15_ph_data),
            ('UDSP', self.l15_ph_data)
        ]

        return record

    @property
    def l15_ph_data_identification(self):

        record = [
            ('Name', (np.str, 30)),
            ('Size', (np.str, 16)),
            ('Address', (np.str, 16))]

        return record


class L15SecondaryProductHeaderRecord(L15PhData):

    '''
    Reference Document:
            MSG Level 1.5 Native Format File Definition
    '''

    def get(self):

        record = [
            ('ABID', self.l15_ph_data),
            ('SMOD', self.l15_ph_data),
            ('APXS', self.l15_ph_data),
            ('AVPA', self.l15_ph_data),
            ('LSCD', self.l15_ph_data),
            ('LMAP', self.l15_ph_data),
            ('QDLC', self.l15_ph_data),
            ('QDLP', self.l15_ph_data),
            ('QQAI', self.l15_ph_data),
            ('SelectedBandIDs', self.l15_ph_data),
            ('SouthLineSelectedRectangle', self.l15_ph_data),
            ('NorthLineSelectedRectangle', self.l15_ph_data),
            ('EastColumnSelectedRectangle', self.l15_ph_data),
            ('WestColumnSelectedRectangle', self.l15_ph_data),
            ('NumberLinesVISIR', self.l15_ph_data),
            ('NumberColumnsVISIR', self.l15_ph_data),
            ('NumberLinesHRV', self.l15_ph_data),
            ('NumberColumnsHRV', self.l15_ph_data)
        ]

        return record


# class GpPkHeaderRecord(object):
#
#    def get(self):
#
#        record = [
#            ('HeaderVersionNo', np.uint8),
#            ('PacketType', np.uint8),
#            ('SubHeaderType', np.uint8),
#            ('SourceFacilityId', np.uint8),
#            ('SourceEnvId', np.uint8),
#            ('SourceInstanceId', np.uint8),
#            ('SourceSUId', np.int32),
#            ('SourceCPUId', (np.uint8, 4)),
#            ('DestFacilityId', np.uint8),
#            ('DestEnvId', np.uint8),
#            ('SequenceCount', np.uint16),
#            ('PacketLength', np.int32)
#        ]
#
#        return record


# class GpPkSh1Record(GSDTRecords):
#
#    def get(self):
#
#        record = [
#            ('SubHeaderVersionNo', np.uint8),
#            ('ChecksumFlag', np.uint8),
#            ('Acknowledgement', (np.uint8, 4)),
#            ('ServiceType', np.uint8),
#            ('ServiceSubtype', np.uint8),
#            ('PacketTime', self.time_cds_short),
#            ('SpacecraftId', self.gp_sc_id)
#        ]
#
#        return record


class L15DataHeaderRecord(GSDTRecords):

    '''
    Reference Document:
            MSG Level 1.5 Image Data Format Description
    '''

    def get(self, umarf=True):

        record = [
            ('15HeaderVersion', np.uint8),
            ('SatelliteStatus', self.satellite_status),
            ('ImageAcquisition', self.image_acquisition),
            ('CelestialEvents', self.celestial_events),
            ('ImageDescription', self.image_description),
            ('RadiometricProcessing', self.radiometric_processing),
            ('GeometricProcessing', self.geometric_processing)]

        if umarf:
            record.append(('IMPFConfiguration', self.impf_configuration))

        return record

    @property
    def satellite_status(self):

        satellite_definition = [
            ('SatelliteId', np.uint16),
            ('NominalLongitude', np.float32),
            ('SatelliteStatus', np.uint8)]

        satellite_operations = [
            ('LastManoeuvreFlag', np.bool),
            ('LastManoeuvreStartTime', self.time_cds_short),
            ('LastManoeuvreEndTime', self.time_cds_short),
            ('LastManoeuvreType', np.uint8),
            ('NextManoeuvreFlag', np.bool),
            ('NextManoeuvreStartTime', self.time_cds_short),
            ('NextManoeuvreEndTime', self.time_cds_short),
            ('NextManoeuvreType', np.uint8)]

        orbit_coeff = [
            ('StartTime', self.time_cds_short),
            ('EndTime', self.time_cds_short),
            ('X', (np.float64, 8)),
            ('Y', (np.float64, 8)),
            ('Z', (np.float64, 8)),
            ('VX', (np.float64, 8)),
            ('VY', (np.float64, 8)),
            ('VZ', (np.float64, 8))]

        orbit = [
            ('PeriodStartTime', self.time_cds_short),
            ('PeriodEndTime', self.time_cds_short),
            ('OrbitPolynomial', (orbit_coeff, 100))]

        attitude_coeff = [
            ('StartTime', self.time_cds_short),
            ('EndTime', self.time_cds_short),
            ('XofSpinAxis', (np.float64, 8)),
            ('YofSpinAxis', (np.float64, 8)),
            ('ZofSpinAxis', (np.float64, 8))]

        attitude = [
            ('PeriodStartTime', self.time_cds_short),
            ('PeriodEndTime', self.time_cds_short),
            ('PrincipleAxisOffsetAngle', np.float64),
            ('AttitudePolynomial', (attitude_coeff, 100))]

        utc_correlation = [
            ('PeriodStartTime', self.time_cds_short),
            ('PeriodEndTime', self.time_cds_short),
            ('OnBoardTimeStart', (np.uint8, 7)),
            ('VarOnBoardTimeStart', np.float64),
            ('A1', np.float64),
            ('VarA1', np.float64),
            ('A2', np.float64),
            ('VarA2', np.float64)]
        record = [
            ('SatelliteDefinition', satellite_definition),
            ('SatelliteOperations', satellite_operations),
            ('Orbit', orbit),
            ('Attitude', attitude),
            ('SpinRetreatRCStart', np.float64),
            ('UTCCorrelation', utc_correlation)]

        return record

    @property
    def image_acquisition(self):

        planned_acquisition_time = [
            ('TrueRepeatCycleStart', self.time_cds_expanded),
            ('PlanForwardScanEnd', self.time_cds_expanded),
            ('PlannedRepeatCycleEnd', self.time_cds_expanded)]

        radiometer_status = [
            ('ChannelStatus', (np.uint8, 12)),
            ('DetectorStatus', (np.uint8, 42))]

        hrv_frame_offsets = [
            ('MDUNomHRVDelay1', np.uint16),
            ('MDUNomHRVDelay2', np.uint16),
            ('Spare', np.uint16),
            ('MDUNomHRVBreakLine', np.uint16)]

        operation_parameters = [
            ('L0_LineCounter', np.uint16),
            ('K1_RetraceLines', np.uint16),
            ('K2_PauseDeciseconds', np.uint16),
            ('K3_RetraceLines', np.uint16),
            ('K4_PauseDeciseconds', np.uint16),
            ('K5_RetraceLines', np.uint16),
            ('XDeepSpaceWindowPosition', np.uint8)]

        radiometer_settings = [
            ('MDUSamplingDelays', (np.uint16, 42)),
            ('HRVFrameOffsets', hrv_frame_offsets),
            ('DHSSSynchSelection', np.uint8),
            ('MDUOutGain', (np.uint16, 42)),
            ('MDUCoarseGain', (np.uint8, 42)),
            ('MDUFineGain', (np.uint16, 42)),
            ('MDUNumericalOffset', (np.uint16, 42)),
            ('PUGain', (np.uint16, 42)),
            ('PUOffset', (np.uint16, 27)),
            ('PUBias', (np.uint16, 15)),
            ('OperationParameters', operation_parameters),
            ('RefocusingLines', np.uint16),
            ('RefocusingDirection', np.uint8),
            ('RefocusingPosition', np.uint16),
            ('ScanRefPosFlag', np.bool),
            ('ScanRefPosNumber', np.uint16),
            ('ScanRefPosVal', np.float32),
            ('ScanFirstLine', np.uint16),
            ('ScanLastLine', np.uint16),
            ('RetraceStartLine', np.uint16)]

        decontamination = [
            ('DecontaminationNow', np.bool),
            ('DecontaminationStart', self.time_cds_short),
            ('DecontaminationEnd', self.time_cds_short)]

        radiometer_operations = [
            ('LastGainChangeFlag', np.bool),
            ('LastGainChangeTime', self.time_cds_short),
            ('Decontamination', decontamination),
            ('BBCalScheduled', np.bool),
            ('BBCalibrationType', np.uint8),
            ('BBFirstLine', np.uint16),
            ('BBLastLine', np.uint16),
            ('ColdFocalPlaneOpTemp', np.uint16),
            ('WarmFocalPlaneOpTemp', np.uint16)]

        record = [
            ('PlannedAcquisitionTime', planned_acquisition_time),
            ('RadiometerStatus', radiometer_status),
            ('RadiometerSettings', radiometer_settings),
            ('RadiometerOperations', radiometer_operations)]

        return record

    @property
    def celestial_events(self):

        earth_moon_sun_coeff = [
            ('StartTime', self.time_cds_short),
            ('EndTime', self.time_cds_short),
            ('AlphaCoef', (np.float64, 8)),
            ('BetaCoef', (np.float64, 8))]

        star_coeff = [
            ('StarId', np.uint16),
            ('StartTime', self.time_cds_short),
            ('EndTime', self.time_cds_short),
            ('AlphaCoef', (np.float64, 8)),
            ('BetaCoef', (np.float64, 8))]

        ephemeris = [
            ('PeriodTimeStart', self.time_cds_short),
            ('PeriodTimeEnd', self.time_cds_short),
            ('RelatedOrbitFileTime', (np.str, 15)),
            ('RelatedAttitudeFileTime', (np.str, 15)),
            ('EarthEphemeris', (earth_moon_sun_coeff, 100)),
            ('MoonEphemeris', (earth_moon_sun_coeff, 100)),
            ('SunEphemeris', (earth_moon_sun_coeff, 100)),
            ('StarEphemeris', (star_coeff, (20, 100)))]
        relation_to_image = [
            ('TypeOfEclipse', np.uint8),
            ('EclipseStartTime', self.time_cds_short),
            ('EclipseEndTime', self.time_cds_short),
            ('VisibleBodiesInImage', np.uint8),
            ('BodiesCloseToFOV', np.uint8),
            ('ImpactOnImageQuality', np.uint8)]

        record = [
            ('CelestialBodiesPosition', ephemeris),
            ('RelationToImage', relation_to_image)]

        return record

    @property
    def image_description(self):

        projection_description = [
            ('TypeOfProjection', np.uint8),
            ('LongitudeOfSSP', np.float32)]

        reference_grid = [
            ('NumberOfLines', np.int32),
            ('NumberOfColumns', np.int32),
            ('LineDirGridStep', np.float32),
            ('ColumnDirGridStep', np.float32),
            ('GridOrigin', np.uint8)]

        planned_coverage_vis_ir = [
            ('SouthernLinePlanned', np.int32),
            ('NorthernLinePlanned', np.int32),
            ('EasternColumnPlanned', np.int32),
            ('WesternColumnPlanned', np.int32)]

        planned_coverage_hrv = [
            ('LowerSouthLinePlanned', np.int32),
            ('LowerNorthLinePlanned', np.int32),
            ('LowerEastColumnPlanned', np.int32),
            ('LowerWestColumnPlanned', np.int32),
            ('UpperSouthLinePlanned', np.int32),
            ('UpperNorthLinePlanned', np.int32),
            ('UpperEastColumnPlanned', np.int32),
            ('UpperWestColumnPlanned', np.int32)]

        level_15_image_production = [
            ('ImageProcDirection', np.uint8),
            ('PixelGenDirection', np.uint8),
            ('PlannedChanProcessing', (np.uint8, 12))]

        record = [
            ('ProjectionDescription', projection_description),
            ('ReferenceGridVIS_IR', reference_grid),
            ('ReferenceGridHRV', reference_grid),
            ('PlannedCoverageVIS_IR', planned_coverage_vis_ir),
            ('PlannedCoverageHRV', planned_coverage_hrv),
            ('Level15ImageProduction', level_15_image_production)]

        return record

    @property
    def radiometric_processing(self):

        rp_summary = [
            ('RadianceLinearization', (np.bool, 12)),
            ('DetectorEqualization', (np.bool, 12)),
            ('OnboardCalibrationResult', (np.bool, 12)),
            ('MPEFCalFeedback', (np.bool, 12)),
            ('MTFAdaptation', (np.bool, 12)),
            ('StrayLightCorrection', (np.bool, 12))]

        level_15_image_calibration = [
            ('CalSlope', np.float64),
            ('CalOffset', np.float64)]

        time_cuc_size = [
            ('CT1', np.uint8),
            ('CT2', np.uint8),
            ('CT3', np.uint8),
            ('CT4', np.uint8),
            ('FT1', np.uint8),
            ('FT2', np.uint8),
            ('FT3', np.uint8)]

        cold_fp_temperature = [
            ('FCUNominalColdFocalPlaneTemp', np.uint16),
            ('FCURedundantColdFocalPlaneTemp', np.uint16)]

        warm_fp_temperature = [
            ('FCUNominalWarmFocalPlaneVHROTemp', np.uint16),
            ('FCURedundantWarmFocalPlaneVHROTemp', np.uint16)]

        scan_mirror_temperature = [
            ('FCUNominalScanMirrorSensor1Temp', np.uint16),
            ('FCURedundantScanMirrorSensor1Temp', np.uint16),
            ('FCUNominalScanMirrorSensor2Temp', np.uint16),
            ('FCURedundantScanMirrorSensor2Temp', np.uint16)]

        m1m2m3_temperature = [
            ('FCUNominalM1MirrorSensor1Temp', np.uint16),
            ('FCURedundantM1MirrorSensor1Temp', np.uint16),
            ('FCUNominalM1MirrorSensor2Temp', np.uint16),
            ('FCURedundantM1MirrorSensor2Temp', np.uint16),
            ('FCUNominalM23AssemblySensor1Temp', np.uint8),
            ('FCURedundantM23AssemblySensor1Temp', np.uint8),
            ('FCUNominalM23AssemblySensor2Temp', np.uint8),
            ('FCURedundantM23AssemblySensor2Temp', np.uint8)]

        baffle_temperature = [
            ('FCUNominalM1BaffleTemp', np.uint16),
            ('FCURedundantM1BaffleTemp', np.uint16)]

        blackbody_temperature = [
            ('FCUNominalBlackBodySensorTemp', np.uint16),
            ('FCURedundantBlackBodySensorTemp', np.uint16)]

        fcu_mode = [
            ('FCUNominalSMMStatus', (np.str, 2)),
            ('FCURedundantSMMStatus', (np.str, 2))]

        extracted_bb_data = [
            ('NumberOfPixelsUsed', np.uint32),
            ('MeanCount', np.float32),
            ('RMS', np.float32),
            ('MaxCount', np.uint16),
            ('MinCount', np.uint16),
            ('BB_Processing_Slope', np.float64),
            ('BB_Processing_Offset', np.float64)]

        bb_related_data = [
            ('OnBoardBBTime', time_cuc_size),
            ('MDUOutGain', (np.uint16, 42)),
            ('MDUCoarseGain', (np.uint8, 42)),
            ('MDUFineGain', (np.uint16, 42)),
            ('MDUNumericalOffset', (np.uint16, 42)),
            ('PUGain', (np.uint16, 42)),
            ('PUOffset', (np.uint16, 27)),
            ('PUBias', (np.uint16, 15)),
            ('DCRValues', (np.uint8, 63)),
            ('X_DeepSpaceWindowPosition', np.int8),
            ('ColdFPTemperature', cold_fp_temperature),
            ('WarmFPTemperature', warm_fp_temperature),
            ('ScanMirrorTemperature', scan_mirror_temperature),
            ('M1M2M3Temperature', m1m2m3_temperature),
            ('BaffleTemperature', baffle_temperature),
            ('BlackBodyTemperature', blackbody_temperature),
            ('FCUMode', fcu_mode),
            ('ExtractedBBData', (extracted_bb_data, 12))]

        black_body_data_used = [
            ('BBObservationUTC', self.time_cds_expanded),
            ('BBRelatedData', bb_related_data)]

        impf_cal_data = [
            ('ImageQualityFlag', np.uint8),
            ('ReferenceDataFlag', np.uint8),
            ('AbsCalMethod', np.uint8),
            ('Pad1', (np.str, 1)),
            ('AbsCalWeightVic', np.float32),
            ('AbsCalWeightXsat', np.float32),
            ('AbsCalCoeff', np.float32),
            ('AbsCalError', np.float32),
            ('GSICSCalCoeff', np.float32),
            ('GSICSCalError', np.float32),
            ('GSICSOffsetCount', np.float32)]

        rad_proc_mtf_adaptation = [
            ('VIS_IRMTFCorrectionE_W', (np.float32, (33, 16))),
            ('VIS_IRMTFCorrectionN_S', (np.float32, (33, 16))),
            ('HRVMTFCorrectionE_W', (np.float32, (9, 16))),
            ('HRVMTFCorrectionN_S', (np.float32, (9, 16))),
            ('StraylightCorrection', (np.float32, (12, 8, 8)))]

        record = [
            ('RPSummary', rp_summary),
            ('Level15ImageCalibration', (level_15_image_calibration, 12)),
            ('BlackBodyDataUsed', black_body_data_used),
            ('MPEFCalFeedback', (impf_cal_data, 12)),
            ('RadTransform', (np.float32, (42, 64))),
            ('RadProcMTFAdaptation', rad_proc_mtf_adaptation)]

        return record

    @property
    def geometric_processing(self):

        opt_axis_distances = [
            ('E-WFocalPlane', (np.float32, 42)),
            ('N_SFocalPlane', (np.float32, 42))]

        earth_model = [
            ('TypeOfEarthModel', np.uint8),
            ('EquatorialRadius', np.float64),
            ('NorthPolarRadius', np.float64),
            ('SouthPolarRadius', np.float64)]

        record = [
            ('OptAxisDistances', opt_axis_distances),
            ('EarthModel', earth_model),
            ('AtmosphericModel', (np.float32, (12, 360))),
            ('ResamplingFunctions', (np.uint8, 12))]

        return record

    @property
    def impf_configuration(self):
        # This needs to be checked further. We believe something is missing
        # after overall configuration

        overall_configuration = [
            ('Issue', np.uint16),
            ('Revision', np.uint16)]

        sw_version = overall_configuration

        info_base_versions = sw_version

        su_configuration = [
            ('SWVersion', sw_version),
            ('InfoBaseVersions', (info_base_versions, 10))]

        su_details = [
            ('SUId', np.uint16),
            ('SUIdInstance', np.uint8),
            ('SUMode', np.uint8),
            ('SUState', np.uint8),
            ('SUConfiguration', su_configuration)]

        equalisation_params = [
            ('ConstCoeff', np.float32),
            ('LinearCoeff', np.float32),
            ('QuadraticCoeff', np.float32)]

        black_body_data_for_warm_start = [
            ('GTotalForMethod1', (np.float64, 12)),
            ('GTotalForMethod2', (np.float64, 12)),
            ('GTotalForMethod3', (np.float64, 12)),
            ('GBackForMethod1', (np.float64, 12)),
            ('GBackForMethod2', (np.float64, 12)),
            ('GBackForMethod3', (np.float64, 12)),
            ('RatioGTotalToGBack', (np.float64, 12)),
            ('GainInFrontOpticsCont', (np.float64, 12)),
            ('CalibrationConstants', (np.float32, 12)),
            ('maxIncidentRadiance', (np.float64, 12)),
            ('TimeOfColdObsSeconds', np.float64),
            ('TimeOfColdObsNanoSecs', np.float64),
            ('IncidenceRadiance', (np.float64, 12)),
            ('TempCal', np.float64),
            ('TempM1', np.float64),
            ('TempScan', np.float64),
            ('TempM1Baf', np.float64),
            ('TempCalSurround', np.float64)]

        mirror_parameters = [
            ('MaxFeedbackVoltage', np.float64),
            ('MinFeedbackVoltage', np.float64),
            ('MirrorSlipEstimate', np.float64)]

        hktm_parameters = [
            ('TimeS0Packet', self.time_cds_short),
            ('TimeS1Packet', self.time_cds_short),
            ('TimeS2Packet', self.time_cds_short),
            ('TimeS3Packet', self.time_cds_short),
            ('TimeS4Packet', self.time_cds_short),
            ('TimeS5Packet', self.time_cds_short),
            ('TimeS6Packet', self.time_cds_short),
            ('TimeS7Packet', self.time_cds_short),
            ('TimeS8Packet', self.time_cds_short),
            ('TimeS9Packet', self.time_cds_short),
            ('TimeSYPacket', self.time_cds_short),
            ('TimePSPacket', self.time_cds_short)]

        warm_start_params = [
            ('ScanningLaw', (np.float64, 1527)),
            ('RadFramesAlignment', (np.float64, 3)),
            ('ScanningLawVariation', (np.float32, 2)),
            ('EqualisationParams', (equalisation_params, 42)),
            ('BlackBodyDataForWarmStart', black_body_data_for_warm_start),
            ('MirrorParameters', mirror_parameters),
            ('LastSpinPeriod', np.float64),
            ('HKTMParameters', hktm_parameters),
            ('WSPReserved', (np.uint8, 3408))]

        record = [
            ('OverallConfiguration', overall_configuration),
            ('SUDetails', (su_details, 50)),
            ('WarmStartParams', warm_start_params),
            ('Dummy', (np.void, 4))]  # FIXME!

        return record
